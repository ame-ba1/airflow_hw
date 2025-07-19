import os
import dill
import pandas as pd
import logging
import json


current_file = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))  # modules → project

# Пути
model_dir = os.path.join(project_root, "data", "models")
test_dir = os.path.join(project_root, "data", "test")
pred_dir = os.path.join(project_root, "data", "predictions")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_latest_model(model_folder: str):

    pkl_files = [f for f in os.listdir(model_folder) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError("Файл модели не найден")
    latest_model = sorted(pkl_files)[-1]
    logging.info(f"Загружается модель: {latest_model}")
    with open(os.path.join(model_folder, latest_model), "rb") as f:
        return dill.load(f)





def load_test_data(test_folder: str) -> list[pd.DataFrame]:

    files = [f for f in os.listdir(test_folder) if f.endswith((".csv", ".json"))]
    if not files:
        raise FileNotFoundError("Файлы для теста не найдены")

    dataframes = []
    for file in files:
        file_path = os.path.join(test_folder, file)

        if file.endswith(".csv"):
            df = pd.read_csv(file_path)

        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)

        else:
            continue

        df["source_file"] = file
        dataframes.append(df)

    return dataframes


def predict():
    model = load_latest_model(model_dir)
    test_dataframes = load_test_data(test_dir)

    all_predictions = []

    for df in test_dataframes:
        predictions = model.predict(df)
        df_result = df.copy()
        df_result["prediction"] = predictions
        all_predictions.append(df_result)

    final_df = pd.concat(all_predictions)

    final_df = final_df[["id", "prediction"]]

    os.makedirs(pred_dir, exist_ok=True)
    output_file = os.path.join(pred_dir, "predictions.csv")
    final_df.to_csv(output_file, index=False)
    logging.info(f"Предсказания сохранены в {output_file}")


if __name__ == "__main__":
    predict()
