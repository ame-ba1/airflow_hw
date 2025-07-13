import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator

# Устанавливаем путь к проекту
project_path = os.path.expanduser('~/airflow_hw')
os.environ['PROJECT_PATH'] = project_path
sys.path.insert(0, project_path)

# Импорт функций после добавления пути
from modules.pipeline import pipeline
from modules.predict import predict

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10, tzinfo=dt.timezone.utc),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
    dag_id='car_price_prediction',
    schedule_interval="0 15 * * *",  # каждый день в 15:00 UTC
    default_args=default_args,
    catchup=False,
    description="Train model and make predictions",
) as dag:

    pipeline_task = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    predict_task = PythonOperator(
        task_id="predict",
        python_callable=predict,
    )

    pipeline_task >> predict_task