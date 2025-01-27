import os
os.environ["KERAS_BACKEND"] = "torch"
import time

import bayesflow as bf
from dl_src.load_data import data_loader
from keras.src.backend.common import global_state
# ensure the backend is set
import torch
import keras


if __name__ == "__main__":

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	global_state.set_global_attribute("torch_device", device)
	# keras.mixed_precision.set_global_policy("mixed_bfloat16")


	train, validation, _ = data_loader()

	summary_dim = 64 # boundary
	num_blocks = 3 
	num_heads = 6 # boundary
	embed_dims = (26,) * num_blocks
	mlp_depths = (2,) * num_blocks
	mlp_widths = (64,) * num_blocks
	summary_dropout = 0.09
	time_embedding = "time2vec"
	time_embed_dim = 16 # boundary
	

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
	inf_width = 477 # boundary
	inf_depth = 2 # left boundary
	inf_dropout = 0.04 # left boundary
	initial_learning_rate = 3.6e-4
	residual = True
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
			verbose=1,
			callbacks=[keras.callbacks.EarlyStopping(patience=5,monitor="val_loss")]
			# callbacks=[KerasPruningCallback(trial, "val_loss", interval=10)]
	)

	
	# 	# Profile training step
	# with torch.profiler.profile(
	# 	activities=[torch.profiler.ProfilerActivity.CPU],
	# 	record_shapes=True,
	# 	profile_memory=True,
	# 	with_stack=True
	# ) as prof:
	start_time = time.time()
	history = approximator.fit(
			epochs=epochs,
			dataset=train,
			validation_data=validation,
			verbose=0
	)
	print("Training time: ", time.time() - start_time)
	# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
