# AI-Based Diabetes Prediction System

A Streamlit academic project that uses a Random Forest classifier to estimate the model's diabetes-risk class from eight input features in the Pima Indians Diabetes dataset.

## Live demo

Try the deployed Streamlit app: **[AI-Based Diabetes Prediction System](https://amit-diabetes-project.streamlit.app/)**

## Tech stack
- Python
- Streamlit
- NumPy and Pandas
- Scikit-learn
- Random Forest Classifier

## Project files
```text
AI-Based-Diabetes-Prediction-System/
├── app.py
├── train_model.py
├── diabetes.csv
├── diabetes_model.pkl
├── requirements.txt
└── README.md
```

## How to use the app
1. Open the [live demo](https://amit-diabetes-project.streamlit.app/).
2. Enter the eight requested health-related input values.
3. Select the prediction button to view the model's estimated probability and predicted class.

## Run locally
Use a Python version supported by the installed Streamlit and Scikit-learn packages.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\\.venv\\Scripts\\Activate.ps1
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model
The training script reads `diabetes.csv` and saves the trained model as `diabetes_model.pkl` in the project root:

```bash
python train_model.py
```

The script prints accuracy on its 20% hold-out split. This is a single split result, not a guarantee of real-world performance.

## Important disclaimer
This is an educational project, not a medical diagnostic tool. Its output is a model estimate and must not replace laboratory testing or advice from a qualified healthcare professional.
