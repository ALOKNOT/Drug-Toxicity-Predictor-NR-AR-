import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def smiles_to_features(smiles, n_bits=1024):
    """Extract Morgan fingerprint features and molecular descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    fp_array = np.array(fp)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    descriptors = np.array([mw, logp, tpsa, hbd, hba])
    return np.hstack([fp_array, descriptors])

def batch_smiles_to_features(smiles_list, n_bits=1024):
    """Convert a list/series of SMILES strings to a feature matrix."""
    features = []
    valid_indices = []
    for idx, sm in enumerate(smiles_list):
        feat = smiles_to_features(sm, n_bits=n_bits)
        if feat is not None:
            features.append(feat)
            valid_indices.append(idx)
    if len(features) == 0:
        return np.empty((0, n_bits + 5)), []
    return np.array(features), valid_indices
