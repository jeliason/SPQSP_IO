import os
# ensure the backend is set
import argparse
import torch
import keras
from torch.utils.data import DataLoader


if __name__ == "__main__":

	# print("Is CUDA available?", torch.cuda.is_available())
	# print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# parser = argparse.ArgumentParser()
	# parser.add_argument("--backend", type=str, default="jax")
	# args = parser.parse_args()
	os.environ["KERAS_BACKEND"] = "torch"

	import bayesflow as bf
	from dl_src.load_data import data_loader
	from keras.src.backend.common import global_state

	# print("Using backend:", args.backend)
	# global_state.set_global_attribute("torch_device", device)


	train, validation, _ = data_loader()

	summary_net = bf.networks.TimeSeriesTransformer(summary_dim=32,
																									time_axis=-1,
																									time_embedding="time2vec")

	inference_net = bf.networks.FlowMatching(
		integrator = "rk2",
			subnet_kwargs={"residual": True, "dropout": 0.0, "widths": (128, 128, 128, 128)}
	)

	# workflow = bf.BasicWorkflow(
	# 		adapter=bf.adapters.Adapter(),
	# 		inference_network=inference_net,
	# 		summary_network=summary_net,
	# 		inference_variables=train.inference_variables
	# )

	# get num_batches from train_loader
	initial_learning_rate = 1e-3
	epochs = 10
	    # Create optimizer
	scheduled_lr = keras.optimizers.schedules.CosineDecay(
			initial_learning_rate=initial_learning_rate,
			decay_steps=epochs*train.num_batches,
			alpha=1e-8
	)
	optimizer = keras.optimizers.Adam(learning_rate=scheduled_lr)
	
	
	# Create approximator
	approximator = bf.ContinuousApproximator(
			summary_network=summary_net,
			inference_network=inference_net,
			adapter=None
	)
	approximator.compile(optimizer=optimizer)
	
		# Profile training step
	with torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU],
		record_shapes=True,
		profile_memory=True,
		with_stack=True
	) as prof:
		# Train and compute the average of last 5 validation losses
		history = approximator.fit(
				epochs=epochs,
				dataset=train,
				validation_data=validation
		)
	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
