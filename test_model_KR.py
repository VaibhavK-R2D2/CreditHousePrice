from data_loader import load_data
import joblib

# Load dataset
df = load_data()

# Features only
X = df.drop("MEDV", axis=1)

# Keep as DataFrame
sample_data = X.iloc[:1]

# Load scaler
scaler = joblib.load(
    "saved_models/scaler.pkl"
)

# Scale data
sample_data = [[
    0.00632,  # CRIM
    18.0,     # ZN
    2.31,     # INDUS
    0,        # CHAS
    0.538,    # NOX
    6.575,    # RM
    65.2,     # AGE
    4.0900,   # DIS
    1,        # RAD
    296,      # TAX
    15.3,     # PTRATIO
    396.90,   # B
    4.98      # LSTAT
]]

sample_data = scaler.transform(sample_data)

# Load Kernel Ridge Regression model
model = joblib.load(
    "saved_models/Kernel Ridge Regressor.pkl"
)

# Predict
prediction = model.predict(sample_data)

print("Kernel Ridge Regression Prediction:")
print(prediction)