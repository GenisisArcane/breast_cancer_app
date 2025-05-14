import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib  # For saving the model and scaler

# Load the breast cancer dataset
data = load_breast_cancer()
breast_cancer_df = pd.DataFrame(data.data, columns=data.feature_names)
breast_cancer_df['target'] = data.target  # Add target column

# Select specific features for analysis
selected_features = [
    'worst radius',
    'worst perimeter',
    'worst concave points',
    'mean concave points',
    'worst area',
    'worst compactness',
    'mean radius',
    'texture error',  # Captures measurement variability
    'worst texture',
    'area error'
]

# Split data into features and label
features = breast_cancer_df[selected_features].copy()
labels = breast_cancer_df["target"].copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split data into train and test sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, labels, train_size=0.7, random_state=0
)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train_scaled, y_train)

# Make predictions
rf_preds = rf.predict(X_test_scaled)
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\nRandom Forest Performance:")
print(classification_report(y_test, rf_preds, target_names=['Malignant', 'Benign']))

# Save the model and scaler for future use
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved as 'random_forest_model.pkl' and 'scaler.pkl'.")