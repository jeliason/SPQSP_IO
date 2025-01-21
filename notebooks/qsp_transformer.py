import os
# ensure the backend is set
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="jax")
parser.add_argument("--where", type=str, default="local")
args = parser.parse_args()

os.environ["KERAS_BACKEND"] = args.backend
HPC_OR_LOCAL = args.where

from dl_src.load_data import data_loader

import bayesflow as bf

train, validation, adapter, inference_variables = data_loader(HPC_OR_LOCAL=HPC_OR_LOCAL)

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
