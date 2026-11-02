from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "diabetes.csv"
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

# Load the dataset stored in the project root.
df = pd.read_csv(DATASET_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
with MODEL_PATH.open("wb") as model_file:
    pickle.dump(model, model_file)

print(f"Model saved to: {MODEL_PATH}")
print(f"Hold-out accuracy on this split: {accuracy:.2%}")
