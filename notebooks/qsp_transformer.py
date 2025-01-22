import os
# ensure the backend is set
import argparse
# import torch


if __name__ == "__main__":

	# print("Is CUDA available?", torch.cuda.is_available())
	# print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# parser = argparse.ArgumentParser()
	# parser.add_argument("--backend", type=str, default="jax")
	# args = parser.parse_args()
	os.environ["KERAS_BACKEND"] = "tensorflow"

	import bayesflow as bf
	from dl_src.load_data import data_loader
	from keras.src.backend.common import global_state

	# print("Using backend:", args.backend)
	# global_state.set_global_attribute("torch_device", device)

	# keras.backend.set_device(device)  # Ensure Keras uses CUDA
	# print("Keras backend is using:", keras.backend.device())


	train, validation, adapter, inference_variables = data_loader()

	summary_net = bf.networks.TimeSeriesTransformer(summary_dim=32,
																									time_axis=-1,
																									time_embedding="time2vec")

	inference_net = bf.networks.FlowMatching(
		integrator = "rk2",
			subnet_kwargs={"residual": True, "dropout": 0.0, "widths": (128, 128, 128, 128)}
	)

	workflow = bf.BasicWorkflow(
			adapter=adapter,
			inference_network=inference_net,
			summary_network=summary_net,
			inference_variables=inference_variables
	)

	history = workflow.fit_offline(train, epochs=100, batch_size=32, validation_data=validation)
