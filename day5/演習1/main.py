import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import random
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature



def prepare_data(test_size=0.2, random_state=42):
    path = "data/Titanic.csv"
    data = pd.read_csv(path)

    # 前処理
    data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    for col in ["Pclass", "Sex", "Age", "Fare", "Survived"]:
        data[col] = data[col].astype(float)

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def log_model(model, accuracy, params, X_train, X_test):
    with mlflow.start_run():
        # パラメータ・スコア
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_metric("accuracy", accuracy)

        # 推論時間
        start = time.time()
        _ = model.predict(X_test)
        elapsed = time.time() - start
        mlflow.log_metric("inference_time", elapsed)
        print(f"Inference time: {elapsed:.4f} seconds")

        # モデル保存
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5]
        )
        print(f"モデルのログ記録値\naccuracy: {accuracy}\nparams: {params}")

    # ローカル保存
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "titanic_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("📦 モデルを models/titanic_model.pkl に保存しました")



if __name__ == "__main__":
    # ランダムハイパーパラメータ設定
    test_size = round(random.uniform(0.1, 0.3), 2)
    data_random_state = random.randint(1, 100)
    model_random_state = random.randint(1, 100)
    n_estimators = random.randint(50, 200)
    max_depth = random.choice([None, 3, 5, 10, 15])

    params = {
        "test_size": test_size,
        "data_random_state": data_random_state,
        "model_random_state": model_random_state,
        "n_estimators": n_estimators,
        "max_depth": "None" if max_depth is None else max_depth,
    }

    # データ準備と学習・評価
    X_train, X_test, y_train, y_test = prepare_data(test_size, data_random_state)
    model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test,
                                         n_estimators, max_depth, model_random_state)

    # モデルログ＋保存
    log_model(model, accuracy, params, X_train, X_test)
