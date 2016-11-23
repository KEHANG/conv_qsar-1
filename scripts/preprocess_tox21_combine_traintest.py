import rdkit.Chem as Chem
import numpy as np
import os

if __name__ == '__main__':

	################################################
	### COMBINE TRAIN AND LEADERBOARD TEST
	################################################

	targets = [
		'NR-AhR',
		'NR-AR',
		'NR-AR-LBD',
		'NR-Aromatase',
		'NR-ER',
		'NR-ER-LBD',
		'NR-PPAR-gamma',
		'SR-ARE',
		'SR-ATAD5',
		'SR-HSE',
		'SR-MMP',
		'SR-p53'
	]

	data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

	import csv
	for target in targets:
		# Write to <target>-traintest.smiles
		with open(os.path.join(data_root, '{}-traintest.smiles'.format(target.lower())), 'w') as out:

			with open(os.path.join(data_root, '{}.smiles'.format(target.lower())), 'rb') as csvfile:
				reader = csv.reader(csvfile, delimiter = '\t')
				for i, row in enumerate(reader):
					out.write('\t'.join(row) + '\n') # copy line to file

			with open(os.path.join(data_root, '{}-test.smiles'.format(target.lower())), 'rb') as csvfile:
				reader = csv.reader(csvfile, delimiter = '\t')
				for i, row in enumerate(reader):
					out.write('\t'.join(row) + '\n') # copy line to file

		print('Merged data for {}'.format(target))