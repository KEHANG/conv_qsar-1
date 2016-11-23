import rdkit.Chem as Chem
import numpy as np
import os

if __name__ == '__main__':

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
	
	################################################
	### EVALUATION DATASET
	################################################

	smiles_codes = os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'tox21_10k_challenge_score.smiles'
	)

	codes_results = os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'tox21_10k_challenge_score.txt'
	)


	# Get SMILES <-> code correspondence
	import csv
	code_dict = {}
	with open(smiles_codes, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = '\t')
		for i, row in enumerate(reader):
			if i == 0: continue
			code_dict[row[1]] = row[0]
			# code_dict[CODE] = SMILES


	mols = []
	smiles = []
	ys = None


	with open(codes_results, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = '\t')
		for i, row in enumerate(reader):
			if i == 0: continue
			smile = code_dict[row[0]]
			mol = Chem.MolFromSmiles(smile, sanitize = False)
			if not mol: 
				print('Could not parse {}'.format(smile))
				raw_input('Pause...')

			mols.append(mol)
			smiles.append(smile)
			y = np.nan * np.ones((1, len(targets)))
			for j, target in enumerate(targets):
				if row[j+1] == 'x': continue
				try:
					y[0, j] = bool(float(row[j+1]))
				except Exception as e:
					print(e)
			if type(ys) == type(None): 
				ys = y
			else:
				ys = np.concatenate((ys, y))
			if i % 500 == 0:
				print('completed {} entries'.format(j))
	
	print(ys)
	print(ys.shape)
	for i, target in enumerate(targets):
		print('Target {} has {} entries; {} active'.format(
			target, sum(~np.isnan(ys[:, i])), np.sum(ys[~np.isnan(ys[:, i]), i])
		))
		with open(os.path.join(
				os.path.dirname(os.path.dirname(__file__)),
				'data', '{}-eval.smiles'.format(target.lower())
				), 'w') as fid:
			for j, smile in enumerate(smiles):
				if ~np.isnan(ys[j, i]):
					fid.write('{}\t{}\t{}\n'.format(smile, '??', ys[j, i]))



	with open(os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'tox21-eval.smiles'
			), 'w') as fid:
		for j, smile in enumerate(smiles):
			fid.write('{}\t{}\t{}\n'.format(smile, '??', '\t'.join([str(x) for x in ys[j, :]])))

