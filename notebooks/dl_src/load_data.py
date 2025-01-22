import os
import numpy as np
import pandas as pd
import bayesflow as bf

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
			parent_dir = './qsp_experiments/'
			exp_dir = parent_dir + 'all_params_10k/outputs/subject_1/'
	else:
			parent_dir = '/nfs/turbo/umms-ukarvind/joelne/SPQSP_IO/qsp_experiments/'
			exp_dir = parent_dir + 'all_params_10k/subject_1/'

	params_df = pd.read_csv(os.path.join(exp_dir,'param_log.csv'), index_col=0, header=0)
	param_names = list(params_df.columns)


	# inference_variables = [
	# 	'QSP/init_value/Parameter/Kd_PD1_PDL1'
	# 	# 'QSP/init_value/Parameter/K_C1_PDLX_Teff_PD1',
	# 	# 'QSP/init_value/Parameter/k_C_growth',
	# 	# 'QSP/init_value/Parameter/n_clone_p1_0'
	# ]
	inference_variables = param_names[9:]

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
	num_samples = int(total_samples * (1 - validation_ratio - test_ratio))
	num_test_samples = int(total_samples * test_ratio)
	total_idx = sorted(qsp_nums)
	# total_idx = np.random.choice(qsp_nums, num_samples + num_test_samples,replace=False)
	data_idx = total_idx[:num_samples]
	test_idx = total_idx[num_samples:(num_samples + num_test_samples)]

	# get text_idx, but exclude the data_idx samples
	qsp_paths = [os.path.join(exp_dir, 'qsp_arr_' + str(i + 1) + '.npz') for i in data_idx]

	observables = [np.load(path)['arr_0'] for path in qsp_paths]
	observables = np.concatenate(observables, axis=0)

	def observables_processor(obs):

		obs = obs[:,:,species_idx]

		obs = obs[:,::thin,:]

		return obs

	data = observables_processor(observables)

	time_linsp = np.linspace(0, 1, data.shape[1])
	time = np.tile(time_linsp, (data.shape[0], 1,1)).transpose((0,2,1))

	def params_processor(params_df,idx):
			params = params_df.to_numpy()[idx]
			return params
	params = params_processor(params_df,data_idx)


	params_idx = [param_names.index(var) for var in inference_variables]

	split = int(validation_ratio * params.shape[0])
	train_params_dict = dict(zip(inference_variables,[params[split:,i,np.newaxis] for i in params_idx]))
	validation_params_dict = dict(zip(inference_variables,[params[:split,i,np.newaxis] for i in params_idx]))

	train = {"sim_data": data[split:],
						"time": time[split:]
						} | train_params_dict
	validation = {"sim_data": data[:split],
								"time": time[:split]
								} | validation_params_dict

	def subsampler(total_time_points,n=10):
		bins = np.linspace(0, total_time_points, n + 1)
		bins = [int(i) for i in bins]
		random_idx = np.random.randint(bins[:-1], bins[1:])
		return random_idx

	def add_noise(arr):
		noise = np.random.normal(0, sigma, arr.shape)
		arr += noise
		arr[arr <= 0] = np.abs(noise[arr <= 0])
		return arr

	adapter = (
			bf.adapters.Adapter()
			.convert_dtype("float64", "float32")
			.as_time_series("sim_data")
			.concatenate(inference_variables, into="inference_variables")
			.rename("sim_data", "summary_variables")
			# since all our variables are non-negative (zero or larger)
			# this .apply call ensures that the variables are transformed
			# to the unconstrained real space and can be back-transformed under the hood
			.apply(exclude=["time"],forward=lambda x: np.log1p(x), inverse=lambda x: np.expm1(x))
			.subsample(["summary_variables","time"],sampler=lambda x: subsampler(x,n=n_time_points))
			.standardize(exclude=["time"])
			# .normalize(exclude=["time"],axis=0)
			# .apply(include=["summary_variables"],forward=lambda x: add_noise(x),inverse=None)
			.concatenate(["summary_variables","time"], into="summary_variables")
			# .drop(["time"])
	)

	return train, validation, adapter, inference_variables
