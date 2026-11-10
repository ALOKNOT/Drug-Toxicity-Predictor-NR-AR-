# Drug-Toxicity-Predictor-NR-AR-

A machine learning application for predicting Nuclear Receptor Androgen Receptor (NR-AR) toxicity of drug candidate compounds based on SMILES input, featuring 2D and 3D molecular visualization.

## Requirements & Environment

- **Tested Python Version:** 3.11

### Dependencies
Dependencies are listed in `requirements.txt`:
- `streamlit==1.66.1`
- `rdkit==2026.9.1`
- `scikit-learn==1.8.0`
- `pandas==3.0.2`
- `numpy==2.4.4`
- `joblib==1.5.3`

Dev Dependencies (`requirements-dev.txt`):
- `pytest==9.2.0`

## Setup & Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Model Training

To train the model on `tox21.csv`:

```bash
python train.py
```

## Running the Web Application

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest
```
