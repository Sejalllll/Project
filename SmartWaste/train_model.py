import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Image: 64x64x3 = 12,288 + 3 sensors = 12,291 total features
n_features = 12291

X = np.random.rand(200, n_features)  # 200 samples with correct feature size
y = np.random.randint(0, 2, 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, 'model/waste_classifier.pkl')
print("✅ Model trained with 12291 features and saved.")
