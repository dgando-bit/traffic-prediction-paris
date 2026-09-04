FROM apache/airflow:3.3.1-python3.12

USER root

COPY packages/shared /opt/airflow/packages/shared
COPY packages/traffic_prediction /opt/airflow/packages/traffic_prediction

RUN chown -R airflow:root /opt/airflow/packages

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-fab \
    /opt/airflow/packages/shared \
    /opt/airflow/packages/traffic_prediction