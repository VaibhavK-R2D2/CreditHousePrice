from data_loader import load_data
from misc import (
    split_data,
    train_model,
    predict,
    evaluate_model,
    get_decision_tree_model
)

import joblib
import os

def main():

    print("Loading dataset...")

    # Load dataset
    df = load_data()

    print("Dataset loaded successfully")

    # Split dataset

    # Preprocess
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    print("Data preprocessing completed")

    # Get models
    model = get_decision_tree_model()
    model_name = f"Decision Tree Regressor"

    # Create folder if not exists
    os.makedirs("saved_models", exist_ok=True)

    # Train and evaluate

    print("\n==========================")
    print(f"Model: {model_name}")
    print("==========================")

    # Train
    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    # Save model
    model_filename = f"saved_models/{model_name}.pkl"

    joblib.dump(
        trained_model,
        model_filename
    )

    print(f"Model saved: {model_filename}")

    # Predict
    predictions = predict(
        trained_model,
        X_test
    )

    # Evaluate
    evaluate_model(
        y_test,
        predictions,
        model_name
    )

    joblib.dump(
        scaler,
        "saved_models/scaler.pkl"
    )


if __name__ == "__main__":
    main()