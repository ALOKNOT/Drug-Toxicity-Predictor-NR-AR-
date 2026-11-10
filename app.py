import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import streamlit.components.v1 as components
from features import smiles_to_features

st.set_page_config(page_title="Drug Toxicity Predictor (NR-AR)", layout="wide")
st.title("Drug Toxicity Predictor - NR-AR (Nuclear Receptor Androgen Receptor)")
st.write("Predict Nuclear Receptor Androgen Receptor (NR-AR) toxicity for small molecules and visualize 2D/3D chemical structures.")

st.sidebar.header("Molecule Input")
smiles_input = st.sidebar.text_input("Enter SMILES string:", value="CC(=O)OC1=CC=CC=C1C(=O)O")

examples = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "Flutamide": "CC(C)C(=O)NC1=CC(=C(C=C1)[N+](=O)[O-])C(F)(F)F",
    "Bicalutamide": "CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O"
}
selected_example = st.sidebar.selectbox("Or choose an example compound:", list(examples.keys()))
if st.sidebar.button("Use Example SMILES"):
    smiles_input = examples[selected_example]

mol = Chem.MolFromSmiles(smiles_input)

if mol is None:
    st.error("Invalid SMILES string. Please enter a valid molecular structure.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2D Structure Viewer")
        img = Draw.MolToImage(mol, size=(400, 400))
        st.image(img, caption="2D Chemical Structure", use_column_width=True)
    with col2:
        st.subheader("3D Structure Viewer")
        mol_3d = Chem.AddHs(mol)
        embed_success = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        if embed_success == 0:
            AllChem.MMFFOptimizeMolecule(mol_3d)
            pdb_block = Chem.MolToPDBBlock(mol_3d)
            html_code = f"""
            <script type="text/javascript" src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <div id="container" style="width: 100%; height: 400px; position: relative;"></div>
            <script>
                let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: 'white'}});
                viewer.addModel(`{pdb_block}`, "pdb");
                viewer.setStyle({{}}, {{stick: {{}}, sphere: {{scale: 0.25}}}});
                viewer.zoomTo();
                viewer.render();
            </script>
            """
            components.html(html_code, height=420)
        else:
            st.warning("Could not generate 3D conformer for this molecule.")

    st.markdown("---")
    st.subheader("NR-AR Toxicity Prediction")
    feat = smiles_to_features(smiles_input)
    if feat is not None:
        model_path = "model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            prob = model.predict_proba([feat])[0][1] if hasattr(model, "predict_proba") else 0.5
            pred = model.predict([feat])[0]
            st.metric(label="Predicted NR-AR Active Status", value="Toxic (Active)" if pred == 1 else "Non-Toxic (Inactive)")
            st.write(f"**Toxicity Probability:** {prob:.2%}")
        else:
            st.info("No pre-trained `model.joblib` found. Run `python train.py` to train the model on `tox21.csv`.")
