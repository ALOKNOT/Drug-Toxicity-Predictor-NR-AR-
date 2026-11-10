import pytest
import numpy as np
from features import smiles_to_features, batch_smiles_to_features

def test_smiles_to_features_valid():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    feat = smiles_to_features(smiles)
    assert feat is not None
    assert len(feat) == 1024 + 5

def test_smiles_to_features_invalid():
    smiles = "InvalidSMILESString123"
    feat = smiles_to_features(smiles)
    assert feat is None

def test_batch_smiles_to_features():
    smiles_list = ["CCO", "InvalidSMILESString123", "CC(=O)O"]
    feats, valid_idx = batch_smiles_to_features(smiles_list)
    assert len(valid_idx) == 2
    assert feats.shape[0] == 2
    assert feats.shape[1] == 1024 + 5
