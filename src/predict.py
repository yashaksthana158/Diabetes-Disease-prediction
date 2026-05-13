"""
predict.py
CLI inference script. Loads the saved model and scaler, accepts patient
clinical values as arguments, and outputs a prediction with probability.

Usage:
    python src/predict.py --glucose 148 --bmi 33.6 --age 50 \
        --pregnancies 6 --blood_pressure 72 --skin_thickness 35 \
        --insulin 0 --dpf 0.627
"""

import argparse
import joblib
import numpy as np


MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURE_ORDER = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Median values from the training set used as defaults for missing inputs
TRAINING_MEDIANS = {
    'Pregnancies': 3.0,
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 23.0,
    'Insulin': 30.5,
    'BMI': 32.0,
    'DiabetesPedigreeFunction': 0.3725,
    'Age': 29.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Prediction — Single Patient Inference")
    parser.add_argument('--pregnancies',    type=float, default=None)
    parser.add_argument('--glucose',        type=float, default=None)
    parser.add_argument('--blood_pressure', type=float, default=None)
    parser.add_argument('--skin_thickness', type=float, default=None)
    parser.add_argument('--insulin',        type=float, default=None)
    parser.add_argument('--bmi',            type=float, default=None)
    parser.add_argument('--dpf',            type=float, default=None,
                        help='DiabetesPedigreeFunction score')
    parser.add_argument('--age',            type=float, default=None)
    return parser.parse_args()


def build_feature_vector(args) -> np.ndarray:
    """Construct the feature array, substituting medians for missing values."""
    arg_map = {
        'Pregnancies':              args.pregnancies,
        'Glucose':                  args.glucose,
        'BloodPressure':            args.blood_pressure,
        'SkinThickness':            args.skin_thickness,
        'Insulin':                  args.insulin,
        'BMI':                      args.bmi,
        'DiabetesPedigreeFunction': args.dpf,
        'Age':                      args.age,
    }
    values = []
    for feat in FEATURE_ORDER:
        val = arg_map[feat]
        if val is None:
            val = TRAINING_MEDIANS[feat]
            print(f"  [info] '{feat}' not provided — using training median ({val})")
        values.append(val)
    return np.array(values).reshape(1, -1)


def predict(features: np.ndarray) -> dict:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "label": "DIABETIC" if prediction == 1 else "NON-DIABETIC",
        "probability": round(float(probability), 4),
    }


def main():
    args = parse_args()
    print("\n--- Diabetes Prediction ---")
    features = build_feature_vector(args)

    result = predict(features)

    print(f"\nResult     : {result['label']}")
    print(f"Probability: {result['probability'] * 100:.1f}% chance of diabetes")

    if result['label'] == "DIABETIC":
        print("\n⚠  High risk detected. Please consult a healthcare professional.")
    else:
        print("\n✔  Low risk detected. Maintain a healthy lifestyle.")


if __name__ == "__main__":
    main()
