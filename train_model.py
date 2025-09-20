import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset load karo
data = pd.read_csv("Crop_recommendation.csv")

# Features (X) aur Target (y) alag karo
X = data.drop("label", axis=1)   # yahan "label" ko apne target column ke naam se replace karna
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train karo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model save karo pickle file me
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")