#!/bin/bash

# === Configuration ===
WORK_DIR="experiments/exp2"                # Folder to create
# IMPORTANT TO CHANGE Params_batch.xml here! n in Params_batch.xml should match the ARRAY_SIZE slurm parameter below
FILES_TO_COPY=("Params_batch.xml" "expBatchGen.py" "make_tensor.R") # Files to copy into the folder

# === Model Configuration ===
RUNTIME=1460                   # Simulation runtime
GRID_INTERVAL=28               # Grid interval
GRID_SAVE=1                    # Save grid (0 for none, 1 for cells only, 2 for substrates, 3 for both)

# === SLURM Configuration ===
TIME="10:00"                # Expected runtime
CPUS_PER_TASK=1                # Number of CPUs per task
ARRAY_SIZE=10                  # Number of jobs in the array
MEM="4G"                       # Memory per node


# === Defaults (likely don't modify) ===
PREPROCESS_SCRIPT_NAME="preprocess.sh"
SLURM_SCRIPT_NAME="run_job.slurm"
POSTPROCESS_SCRIPT_NAME="postprocess.sh"
MASTER_SCRIPT_NAME="run_all.sh"
SBATCH_LOG="job_output.log"
PREPROCESS_LOG="preprocess.log"
POSTPROCESS_LOG="postprocess.log"
EXEC_FOLDER="../SPQSP_IO/NSCLC/NSCLC_multi/macos"  # Folder containing the executable
EXEC_NAME="nsclc_sim_multi"  # Executable name


if [ "$SYSTEM_ENV" != "laptop" ]; then
	VENV_ACTIVATE="source ~/virtual_envs/physicell/bin/activate"
    OUT_FOLDER="/nfs/turbo/umms-ukarvind/joelne/SPQSP_IO/$WORK_DIR"  # Output folder
else
    VENV_ACTIVATE="mamba activate spqsp"
    OUT_FOLDER="vp"  # Output folder
fi


# === Step 1: Create the folder ===
mkdir -p "$WORK_DIR"
echo "Created folder: $WORK_DIR"

# === Step 2: Copy files to the folder ===
for file in "${FILES_TO_COPY[@]}"; do
    if [[ -f "$file" ]]; then
        cp "$file" "$WORK_DIR/"
        echo "Copied $file to $WORK_DIR"
    else
        echo "Warning: $file does not exist, skipping."
    fi
done

# === Step 3: Create preprocessing script ===
cat << EOF > "$WORK_DIR/$PREPROCESS_SCRIPT_NAME"
#!/bin/bash
echo "Preprocessing started" > $PREPROCESS_LOG
cd $EXEC_FOLDER
make $EXEC_NAME

cd ../../../../$WORK_DIR
mkdir -p $OUT_FOLDER

$VENV_ACTIVATE
python expBatchGen.py Params_batch.xml $OUT_FOLDER

echo "Preprocessing done" >> $PREPROCESS_LOG
EOF
chmod +x "$WORK_DIR/$PREPROCESS_SCRIPT_NAME"
echo "Created preprocessing script: $WORK_DIR/$PREPROCESS_SCRIPT_NAME"

# === Step 4: Create run_simulation.sh script ===
cat << EOF > "$WORK_DIR/run_simulation.sh"
#!/bin/bash
ID=\$1
EXEC="$EXEC_FOLDER/$EXEC_NAME"
PARAMS_FILE="$OUT_FOLDER/subject_1/sample_\${ID}/param_1_\${ID}_1.xml"
OUT_FILE="$OUT_FOLDER/subject_1/sample_\${ID}"

# run simulation
\${EXEC} -p \${PARAMS_FILE} -o \${OUT_FILE} -t $RUNTIME -S -G $GRID_SAVE --grid-interval $GRID_INTERVAL

# run make_tensor
Rscript make_tensor.R $OUT_FOLDER/subject_1 \${ID}
EOF
# === Step 4: Create SLURM submission script ===
cat << EOF > "$WORK_DIR/$SLURM_SCRIPT_NAME"
#!/bin/bash
#SBATCH --job-name=$WORK_DIR
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --error=logs/error_%A_%a.log
#SBATCH --time=$TIME
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --array=1-$ARRAY_SIZE  # Example: SLURM array with 10 jobs. Important that index starts from 1!

echo "Running SLURM job \${SLURM_ARRAY_TASK_ID} on \$(hostname)"
ID=\$SLURM_ARRAY_TASK_ID
# load modules
module load Rgeospatial

# run simulation
bash run_simulation.sh \${ID}

echo "SLURM job \${SLURM_ARRAY_TASK_ID} done"
EOF
chmod +x "$WORK_DIR/$SLURM_SCRIPT_NAME"
echo "Created SLURM submission script: $WORK_DIR/$SLURM_SCRIPT_NAME"

# === Step 5: Create postprocessing script ===
cat << EOF > "$WORK_DIR/$POSTPROCESS_SCRIPT_NAME"
#!/bin/bash
echo "Postprocessing started" > $POSTPROCESS_LOG
echo "Postprocessing done" >> $POSTPROCESS_LOG
EOF
chmod +x "$WORK_DIR/$POSTPROCESS_SCRIPT_NAME"
echo "Created postprocessing script: $WORK_DIR/$POSTPROCESS_SCRIPT_NAME"

# === Step 6: Create master script ===
cat << EOF > "$WORK_DIR/$MASTER_SCRIPT_NAME"
#!/bin/bash

# Run preprocessing
./$PREPROCESS_SCRIPT_NAME
if [[ \$? -ne 0 ]]; then
    echo "Preprocessing failed. Exiting."
    exit 1
fi

# Submit SLURM job
JOB_ID=\$(sbatch $SLURM_SCRIPT_NAME | awk '{print \$4}')
echo "Submitted SLURM job array with Job ID: \$JOB_ID"

# Wait for SLURM job to complete
echo "Waiting for SLURM job array \$JOB_ID to finish..."
while squeue -j "\$JOB_ID" > /dev/null 2>&1; do
    sleep 5
done
echo "SLURM job array \$JOB_ID has completed."

# Run postprocessing
./$POSTPROCESS_SCRIPT_NAME
if [[ \$? -ne 0 ]]; then
    echo "Postprocessing failed."
    exit 1
fi
EOF
chmod +x "$WORK_DIR/$MASTER_SCRIPT_NAME"
echo "Created master script: $WORK_DIR/$MASTER_SCRIPT_NAME"

echo "Workflow setup complete. Folder $WORK_DIR is ready for transfer."
