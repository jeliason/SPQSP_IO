import os
# ensure the backend is set
import argparse


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--backend", type=str, default="jax")
	parser.add_argument("--device", type=str, default="cpu")
	args = parser.parse_args()
	os.environ["KERAS_BACKEND"] = args.backend

	import bayesflow as bf
	from dl_src.load_data import data_loader
	from keras.src.backend.common import global_state

	global_state.set_global_attribute("torch_device", args.device)


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
