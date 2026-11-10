import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import batch_smiles_to_features

def train_model(data_path="tox21.csv", model_out="model.joblib"):
    df = pd.read_csv(data_path)
    smiles_col = 'smiles' if 'smiles' in df.columns else ('SMILES' if 'SMILES' in df.columns else df.columns[0])
    target_col = 'NR-AR' if 'NR-AR' in df.columns else df.columns[1]
    df = df.dropna(subset=[smiles_col, target_col])
    X, valid_idx = batch_smiles_to_features(df[smiles_col].tolist())
    y = df[target_col].iloc[valid_idx].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model Training Complete. Test Accuracy: {acc:.4f}")
    joblib.dump(clf, model_out)
    return clf

if __name__ == "__main__":
    train_model()
