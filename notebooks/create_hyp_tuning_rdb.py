import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

study_name = "study_rmse_calerror"  # Unique identifier of the study.

storage = JournalStorage(JournalFileBackend(f"{study_name}.log"))

study = optuna.create_study(study_name=study_name,
														storage=storage,
														directions=["minimize", "minimize"])
