from __future__ import print_function
import os
from qsar.main.data import get_data_full

if __name__ == '__main__':
	
	# Get *FULL* dataset
	data_kwargs = {}
	data_kwargs['batch_size'] = 1
	data_kwargs['training_ratio'] = 1.0
	data_kwargs['cv_folds'] = '1/1'
	data_kwargs['shuffle_seed'] = 0

	##############################
	### DEFINE TESTING CONDITIONS
	##############################

	datasets = [
		'abraham',
		'delaney',
		'bradley',
	]

	for j, dataset in enumerate(datasets):
		print('DATASET: {}'.format(dataset))

		data_kwargs['data_label'] = dataset
		data = get_data_full(**data_kwargs)[0] # training set contains it all

		# Save
		with open(os.path.join(os.path.dirname(__file__), 'coley_' + dataset + '.tdf'), 'w') as fid:
			fid.write('SMILES\t' + data['y_label'] + '\n')
			for j in range(len(data['y'])): # for each mol in that list
				fid.write('{}\t{}\n'.format(data['smiles'][j], data['y'][j]))