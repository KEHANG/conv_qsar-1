import rdkit.Chem as Chem
import numpy as np
import os

if __name__ == '__main__':

	# Read SDF
	suppl = Chem.SDMolSupplier(
		os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'tox21_10k_challenge_test.sdf'
		),
		sanitize = False
	)

	mols = []
	smiles = []
	ys = None
	targets = [
		'NR-AR',
		'NR-AhR',
		'NR-AR-LBD',
		'NR-ER',
		'NR-ER-LBD',
		'NR-aromatase',
		'NR-PPAR-gamma',
		'SR-ARE',
		'SR-ATAD5',
		'SR-HSE',
		'SR-MMP',
		'SR-p53'
	]
	j = 1
	for mol in suppl:
		mols.append(mol)
		smiles.append(Chem.MolToSmiles(mol))
		y = np.nan * np.ones((1, len(targets)))
		for i, target in enumerate(targets):
			try:
				y[0, i] = bool(float(mol.GetProp(target)))
			except Exception as e:
				pass
		if type(ys) == type(None): 
			ys = y
		else:
			ys = np.concatenate((ys, y))
		if j % 500 == 0:
			print('completed {} entries'.format(j))
		j += 1
	
	print(ys)
	print(ys.shape)
	for i, target in enumerate(targets):
		print('Target {} has {} entries; {} active'.format(
			target, sum(~np.isnan(ys[:, i])), np.sum(ys[~np.isnan(ys[:, i]), i])
		))
		with open(os.path.join(
				os.path.dirname(os.path.dirname(__file__)),
				'data', '{}-test.smiles'.format(target.lower())
				), 'w') as fid:
			for j, smile in enumerate(smiles):
				if ~np.isnan(ys[j, i]):
					fid.write('{}\t{}\t{}\n'.format(smile, '??', ys[j, i]))