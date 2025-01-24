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

	initial_learning_rate = 1e-3
	epochs = 1
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
