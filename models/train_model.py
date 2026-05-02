import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os
from pathlib import Path

def load_data(path=r"D:/credit-card-ml-deployment/credit-card-ml-deployment/UCI_Credit_Card.csv"):
    df = pd.read_csv(path)
    y = df["default.payment.next.month"]
    X = df.drop(columns=["ID", "default.payment.next.month"])
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Разделяем признаки на категориальные (номинальные) и числовые
    nominal_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [col for col in X.columns if col not in nominal_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), nominal_features)
        ],
        remainder="passthrough"
    )

    # Модель v1: логистическая регрессия с балансировкой
    model_v1 = LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        random_state=42
    )
    # Модель v2: случайный лес с балансировкой
    model_v2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )

    # Применяем препроцессор
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Обучение
    model_v1.fit(X_train_prep, y_train)
    model_v2.fit(X_train_prep, y_train)

    # Оценка
    print("=== Logistic Regression (v1) ===")
    print(classification_report(y_test, model_v1.predict(X_test_prep)))

    print("\n=== Random Forest (v2) ===")
    print(classification_report(y_test, model_v2.predict(X_test_prep)))

    # Определяем корень проекта
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Путь к папке models внутри корня проекта
    MODELS_DIR = PROJECT_ROOT / "models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Сохраняем модели и препроцессор
    with open(MODELS_DIR / "model_v1.pkl", "wb") as f:
        pickle.dump(model_v1, f)
    with open(MODELS_DIR / "model_v2.pkl", "wb") as f:
        pickle.dump(model_v2, f)
    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Сохраняем имена признаков
    feature_names = numeric_features + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(nominal_features)
    )
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print(f"Модели сохранены в {MODELS_DIR}")


if __name__ == "__main__":
    main()