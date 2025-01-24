import optuna
import optuna_distributed
import logging
import sys
import argparse
import os

study_name = "study_loss_calerror"  # Unique identifier of the study.

def objective(trial, epochs=75):
		import os
		# ensure the backend is set
		if "KERAS_BACKEND" not in os.environ:
				# set this to "torch", "tensorflow", or "jax"
				os.environ["KERAS_BACKEND"] = "torch"
		
		# from optuna.integration import KerasPruningCallback

		import numpy as np
		import torch
		from keras.src.backend.common import global_state

		import keras

		import bayesflow as bf

		from dl_src.load_data import data_loader
			

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		global_state.set_global_attribute("torch_device", device)
		
		# Load data
		print("Loading data...")
		train_dataset, val_dataset, _ = data_loader()
		print("Data loaded.")


		summary_dim = trial.suggest_int("summary_dim", 8, 64)
		num_blocks = trial.suggest_int("num_blocks", 1, 4)
		num_heads = (trial.suggest_int("num_heads", 2, 6),) * num_blocks
		embed_dims = (trial.suggest_int("embed_dims", 8, 128),) * num_blocks
		mlp_depths = (trial.suggest_int("mlp_depths", 1, 4),) * num_blocks
		mlp_widths = (trial.suggest_int("mlp_widths", 16, 256),) * num_blocks
		summary_dropout = trial.suggest_float("mlp_dropout", 0.01, 0.5)
		time_embedding = trial.suggest_categorical("time_embedding", ["time2vec", "lstm", "gru"])
		time_embed_dim = trial.suggest_int("time_embed_dim", 4,16)

		summary_net = bf.networks.TimeSeriesTransformer(
			summary_dim=summary_dim,
			embed_dims=embed_dims,
			num_heads=num_heads,
			mlp_depths=mlp_depths,
			mlp_widths=mlp_widths,
			dropout=summary_dropout,
			time_axis=-1,
			time_embedding=time_embedding,
			time_embed_dim=time_embed_dim
		)
		
		# Optimize hyperparameters
		inf_width = trial.suggest_int("width", 128, 512)
		inf_depth = trial.suggest_int("depth", 2, 8)
		inf_dropout = trial.suggest_float("dropout", 0.01, 0.5)
		initial_learning_rate = trial.suggest_float("lr", 1e-4, 1e-3)
		residual = trial.suggest_categorical("residual", [True, False])
		# spectral_normalization = trial.suggest_categorical("spectral_normalization", [True, False])
		
		# Create inference net
		sigma2 = 1
		inference_network = bf.networks.ContinuousConsistencyModel(
				subnet_kwargs={
					"widths": (inf_width,)*inf_depth,
				 "dropout": inf_dropout, 
				 "residual": residual
				#  "spectral_normalization": spectral_normalization
				},
				sigma_data=sigma2
		)
		
		# Create optimizer
		scheduled_lr = keras.optimizers.schedules.CosineDecay(
				initial_learning_rate=initial_learning_rate,
				decay_steps=epochs*train_dataset.num_batches,
				alpha=1e-8
		)
		optimizer = keras.optimizers.Adam(learning_rate=scheduled_lr)
		
		
		# Create approximator
		approximator = bf.ContinuousApproximator(
			summary_network=summary_net,
			inference_network=inference_network,
			adapter=None
		)
		approximator.compile(optimizer=optimizer)
		
		# Train and compute the average of last 10 validation losses
		history = approximator.fit(
				epochs=epochs,
				dataset=train_dataset,
				validation_data=val_dataset,
				verbose=1
				# callbacks=[KerasPruningCallback(trial, "val_loss", interval=10)]
		)
		loss = np.mean(history.history["val_loss"][-10:])

		summaries = []
		references = []
		for i in range(val_dataset.num_batches):
			batch = val_dataset[i]
			summaries.append(batch["summary_variables"])
			references.append(batch["inference_variables"])

		summaries = torch.cat(summaries, dim=0)
		references = torch.cat(references, dim=0)

		targets = approximator._sample(num_samples=500,summary_variables=summaries)

		cal_dict = bf.diagnostics.metrics.calibration_error(targets.numpy(),references.numpy())
		cal_error = np.mean(cal_dict["values"])

		return loss, cal_error


if __name__ == "__main__":
		# By default, we are relying on process based parallelism to run
		# all trials on a single machine. However, with Dask client, we can easily scale up
		# to Dask cluster spanning multiple physical workers. To learn how to setup and use
		# Dask cluster, please refer to https://docs.dask.org/en/stable/deploying.html.
		SYSTEM_ENV = os.environ.get('SYSTEM_ENV')
		if SYSTEM_ENV == "HPC":
			from dask_jobqueue.slurm import SLURMCluster
			job_script_prologue = ['source ~/virtual_envs/bayesflow/bin/activate',
													'cd ~/repositories/SPQSP_IO/notebooks',
													'echo "Activated Virtual Environment: $VIRTUAL_ENV"',
													'echo "Current Working Directory: $(pwd)"']
			cluster = SLURMCluster(
				account = "ukarvind0",
				cores=1,
				memory="16G",
				walltime="1:00:00",
				job_script_prologue=job_script_prologue,
				log_directory="logs",
				worker_extra_args=["--lifetime", "55m", "--lifetime-stagger", "4m"],
			)
			cluster.adapt(minimum=1, maximum=20)  # Tells Dask to call `srun -n 1 ...` when it needs new workers
			from dask.distributed import Client
			client = Client(cluster)
		else:
			client = None

		parser = argparse.ArgumentParser()
		parser.add_argument("--n_trials", type=int, default=100)

		args = parser.parse_args()
		n_trials = args.n_trials

		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
		storage_name = "sqlite:///{}.db".format(study_name)

		# Optuna-distributed just wraps standard Optuna study. The resulting object behaves
		# just like regular study, but optimization process is asynchronous.

		study = optuna_distributed.from_study(optuna.create_study(study_name=study_name,
																														storage=storage_name,
																														load_if_exists=True,
																														directions=["minimize","minimize"]), client=client)

		# And let's continue with original Optuna example from here.
		# Let us minimize the objective function above.
		# objective = partial(objective, epochs=1)

		print(f"Running {n_trials} trials...")
		study.optimize(objective, n_trials=n_trials, n_jobs=1)
