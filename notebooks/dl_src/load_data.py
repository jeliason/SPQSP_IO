import os
import numpy as np
import pandas as pd
import torch

import keras
import numpy as np

from pyprojroot import here



class OfflineQSPDataset(keras.utils.PyDataset):
		"""
		A dataset that is pre-simulated and stored in memory.
		"""

		def __init__(self, data, params_df, batch_size: int, thin=4, species_idx=list(range(17)), n_bins=10, sigma=1e-2, **kwargs):
				super().__init__(**kwargs)
				self.batch_size = batch_size
				self.data = torch.tensor(data, dtype=torch.float32)
				self.num_samples = data.shape[0]
				self.indices = np.arange(self.num_samples, dtype="int64")

				# create time array
				time_linsp = np.linspace(0, 1, data.shape[1])
				self.time = torch.tensor(np.tile(time_linsp, (data.shape[0], 1,1)).transpose((0,2,1)).astype(np.float32))

				# get parameters
				param_names = list(params_df.columns)
				params = params_df.to_numpy()
				self.inference_variables = param_names[9:]
				self.inference_variables = [self.inference_variables[i] for i in [6,7,8,10,11,12,13]]
				params_idx = [param_names.index(var) for var in self.inference_variables]
				self.params = torch.tensor(np.array([params[:,i] for i in params_idx]).astype(np.float32).T)

				# extract measured species
				self.data = self.data[:,:,species_idx]

				# pre-thin data
				self.data = self.data[:,::thin,:]

				# log1p transform data and params
				self.data = torch.log1p(self.data)
				self.params = torch.log1p(self.params)

				# standardize params
				self.params_mean = torch.mean(self.params, axis=0)
				self.params_std = torch.std(self.params, axis=0)
				self.params = (self.params - self.params_mean) / self.params_std

				# create subsampling bins
				total_time_points = self.data.shape[1]
				bins = np.linspace(0, total_time_points, n_bins + 1)
				self.subsamp_bins = [int(i) for i in bins]

				# sigma for noise
				self.sigma = sigma

				self.shuffle()

		def __getitem__(self, item: int) -> dict[str, np.ndarray]:
				"""Get a batch of pre-simulated data"""
				if not 0 <= item < self.num_batches:
						raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

				item = slice(item * self.batch_size, (item + 1) * self.batch_size)
				item = self.indices[item]

				sample = self.data[item]
				time = self.time[item]
				label = self.params[item]

				# subsample
				random_idx = np.random.randint(self.subsamp_bins[:-1], self.subsamp_bins[1:])
				sample = sample[:,random_idx,:]
				time = time[:,random_idx,:]

				# standardize data
				sample = (sample - torch.mean(sample)) / torch.std(sample)

				# create torch normal noise
				noise = torch.normal(0, self.sigma, sample.shape)
				sample += noise
				sample[sample <= 0] = torch.abs(noise[sample <= 0])

				# concatenate time
				sample = torch.cat([sample, time], axis=-1)

				batch = {
					"summary_variables": sample,
					"inference_variables": label
				}

				return batch

		@property
		def num_batches(self) -> int | None:
				return int(np.ceil(self.num_samples / self.batch_size))

		def on_epoch_end(self) -> None:
				self.shuffle()

		def shuffle(self) -> None:
				"""Shuffle the dataset in-place."""
				np.random.shuffle(self.indices)


def data_loader(
		total_samples = 10000,
		thin = 4,
		validation_ratio = 0.1,
		test_ratio = 0.1,
		sigma = 1e-2,
		n_time_points = 10,
		species_idx = list(range(17))			
):

	# ------ Set parameters ------
	HPC_OR_LOCAL = os.environ.get('SYSTEM_ENV')
	if HPC_OR_LOCAL == 'laptop':
			proj_root = here()
			parent_dir = os.path.join(proj_root,'qsp_experiments/')
			exp_dir = parent_dir + 'all_params_10k/outputs/subject_1/'
	else:
			parent_dir = '/nfs/turbo/umms-ukarvind/joelne/SPQSP_IO/qsp_experiments/'
			exp_dir = parent_dir + 'all_params_10k/subject_1/'

	params_df = pd.read_csv(os.path.join(exp_dir,'param_log.csv'), index_col=0, header=0)


	# inference_variables = [
	# 	'QSP/init_value/Parameter/Kd_PD1_PDL1'
	# 	# 'QSP/init_value/Parameter/K_C1_PDLX_Teff_PD1',
	# 	# 'QSP/init_value/Parameter/k_C_growth',
	# 	# 'QSP/init_value/Parameter/n_clone_p1_0'
	# ]

	species_to_keep = [
		# 'time',
		# blood species
		'Cent.Teff_1_0',
		'Cent.Treg',
		'Cent.nT_CD4',
		'Cent.nT_CD8',
		'Cent.Nivo',
		# lymph node species
		'LN.Nivo',
		'LN.APC',
		'LN.mAPC',
		'LN.nT_CD8',
		'LN.nT_CD4',
		'LN.Treg',
		'LN.aTreg_CD4',
		'LN.IL2',
		'LN.Cp',
		'LN.D1_0',
		'LN.aT_1_0',
		'LN.Teff_1_0',
		# tumor species
		'Tum.Nivo',
		'Tum.APC',
		'Tum.mAPC',
		'Tum.C1',
		'Tum.Cp',
		'Tum.Ckine_Mat',
		'Tum.Treg',
		'Tum.Teff_PD1',
		'Tum.D1_0',
		'Tum.Teff_1_0',
		'Tum.Teff_exhausted',
		'Tum.DAMP',
		'Tum.C1_PDL1',
		'Tum.C1_PDL2'
	]

	# species_idx += [24,29] # include some tumor samples as well
	# species_idx = range(len(species_to_keep))

	# get all qsp paths ending in .npz
	qsp_files = [f for f in os.listdir(exp_dir) if f.endswith('.npz')]
	# extract the qsp number from the file name
	qsp_nums = [int(q.split('_')[2].split('.')[0])-1 for q in qsp_files]
	total_files = len(qsp_files)
	if total_samples > total_files:
		total_samples = total_files
	num_train_samples = int(total_samples * (1 - validation_ratio - test_ratio))
	num_val_samples = int(total_samples * validation_ratio)
	total_idx = sorted(qsp_nums)
	params_df = params_df.iloc[total_idx].reset_index(drop=True)


	# get text_idx, but exclude the data_idx samples
	qsp_paths = [os.path.join(exp_dir, 'qsp_arr_' + str(i + 1) + '.npz') for i in total_idx]
	obs = [np.load(path)['arr_0'] for path in qsp_paths]
	obs = np.concatenate(obs, axis=0)

	train_dataset = OfflineQSPDataset(obs[:num_train_samples], params_df.iloc[:num_train_samples],batch_size=32)
	validation_dataset = OfflineQSPDataset(obs[num_train_samples:num_train_samples+num_val_samples], params_df.iloc[num_train_samples:num_train_samples+num_val_samples],batch_size=32)

	# adapter = (
	# 		bf.adapters.Adapter()
	# 		# .convert_dtype("float64", "float32")
	# 		# .as_time_series("sim_data")
	# 		.concatenate(inference_variables, into="inference_variables")
	# 		.rename("sim_data", "summary_variables")
	# 		# since all our variables are non-negative (zero or larger)
	# 		# this .apply call ensures that the variables are transformed
	# 		# to the unconstrained real space and can be back-transformed under the hood
	# 		.apply(exclude=["time"],forward=lambda x: np.log1p(x), inverse=lambda x: np.expm1(x))
	# 		.subsample(["summary_variables","time"],sampler=lambda x: subsampler(x,n=n_time_points))
	# 		.standardize(exclude=["time"])
	# 		# .normalize(exclude=["time"],axis=0)
	# 		# .apply(include=["summary_variables"],forward=lambda x: add_noise(x),inverse=None)
	# 		.concatenate(["summary_variables","time"], into="summary_variables")
	# 		# .drop(["time"])
	# )

	return train_dataset, validation_dataset, train_dataset.inference_variables
