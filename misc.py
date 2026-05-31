import numpy as np

from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

def get_decision_tree_model():  

    model = DecisionTreeRegressor(
        random_state=42
    )

    return model

def split_data(df):

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model


def predict(model, X_test):

    return model.predict(X_test)


def evaluate_model(y_test, predictions, model_name="Model"):

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    result = f"""
==========================
Model: {model_name}
==========================
MAE  : {mae:.2f}
MSE  : {mse:.2f}
RMSE : {rmse:.2f}
R2   : {r2:.2f}


Average MSE Score: {mse:.4f}


"""

    print(result)

    # Save to file
    with open("results/metrics.txt", "a") as file:
        file.write(result)