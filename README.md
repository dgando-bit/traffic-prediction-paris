### Arborescence cible finale du projet
```
traffic-prediction-paris/
│
├── README.md
├── LICENSE
├── Makefile
├── compose.yaml
├── compose.override.yaml
├── .env.example
├── .gitignore
├── .dockerignore
│
├── config/
│   ├── base.yaml
│   ├── development.yaml
│   └── production.yaml
│
├── data/
│   ├── raw/
│   │   ├── traffic/
│   │   │   └── .gitkeep
│   │   ├── roads/
│   │   │   └── .gitkeep
│   │   ├── weather/
│   │   │   └── .gitkeep
│   │   ├── holidays/
│   │   │   └── .gitkeep
│   │   ├── school_holidays/
│   │   │   └── .gitkeep
│   │   └── events/
│   │       └── .gitkeep
│   │
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── predictions/
│       └── .gitkeep
│
├── packages/
│   │
│   ├── shared/
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   └── shared/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── logging.py
│   │       ├── exceptions.py
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── dates.py
│   │           └── paths.py
│   │
│   └── traffic_prediction/
│       ├── pyproject.toml
│       ├── README.md
│       └── traffic_prediction/
│           ├── __init__.py
│           │
│           ├── ingestion/
│           │   ├── __init__.py
│           │   ├── traffic.py
│           │   ├── roads.py
│           │   ├── weather.py
│           │   ├── holidays.py
│           │   ├── school_holidays.py
│           │   └── events.py
│           │
│           ├── processing/
│           │   ├── __init__.py
│           │   ├── cleaning.py
│           │   ├── validation.py
│           │   └── joins.py
│           │
│           ├── features/
│           │   ├── __init__.py
│           │   ├── traffic_features.py
│           │   ├── time_features.py
│           │   ├── weather_features.py
│           │   ├── calendar_features.py
│           │   ├── event_features.py
│           │   └── build_features.py
│           │
│           ├── models/
│           │   ├── __init__.py
│           │   ├── baseline.py
│           │   ├── train.py
│           │   ├── evaluate.py
│           │   ├── predict.py
│           │   └── registry.py
│           │
│           ├── monitoring/
│           │   ├── __init__.py
│           │   ├── data_quality.py
│           │   ├── data_drift.py
│           │   └── model_metrics.py
│           │
│           └── utils/
│               ├── __init__.py
│               └── congestion.py
│
├── services/
│   │
│   ├── api/
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── schemas.py
│   │       ├── dependencies.py
│   │       └── routes/
│   │           ├── __init__.py
│   │           ├── health.py
│   │           ├── predictions.py
│   │           └── map.py
│   │
│   ├── airflow/
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── dags/
│   │       ├── traffic_ingestion.py
│   │       ├── weather_ingestion.py
│   │       ├── calendar_ingestion.py
│   │       ├── events_ingestion.py
│   │       ├── feature_pipeline.py
│   │       ├── training_pipeline.py
│   │       └── prediction_pipeline.py
│   │
│   ├── training/
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── train.py
│   │       ├── evaluate.py
│   │       └── register.py
│   │
│   └── dashboard/
│       ├── pyproject.toml
│       ├── Dockerfile
│       ├── README.md
│       └── app/
│           ├── main.py
│           ├── pages/
│           │   ├── traffic_map.py
│           │   ├── road_details.py
│           │   └── model_performance.py
│           └── components/
│               ├── map.py
│               ├── charts.py
│               └── metrics.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_traffic_analysis.ipynb
│   ├── 03_weather_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_experiments.ipynb
│
├── scripts/
│   ├── download_initial_data.py
│   ├── build_dataset.py
│   ├── train_model.py
│   └── predict.py
│
├── tests/
│   ├── conftest.py
│   │
│   ├── unit/
│   │   ├── shared/
│   │   │   ├── test_config.py
│   │   │   └── test_dates.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── test_traffic.py
│   │   │   ├── test_weather.py
│   │   │   └── test_events.py
│   │   │
│   │   ├── processing/
│   │   │   ├── test_cleaning.py
│   │   │   └── test_validation.py
│   │   │
│   │   ├── features/
│   │   │   ├── test_traffic_features.py
│   │   │   ├── test_time_features.py
│   │   │   └── test_calendar_features.py
│   │   │
│   │   └── models/
│   │       ├── test_baseline.py
│   │       └── test_predict.py
│   │
│   ├── integration/
│   │   ├── test_traffic_pipeline.py
│   │   ├── test_training_pipeline.py
│   │   └── test_api.py
│   │
│   └── data/
│       └── sample_traffic.parquet
│
├── deployments/
│   ├── nginx/
│   │   ├── nginx.conf
│   │   └── conf.d/
│   │       └── traffic-prediction.conf
│   │
│   ├── prometheus/
│   │   └── prometheus.yml
│   │
│   └── grafana/
│       ├── dashboards/
│       │   ├── api-monitoring.json
│       │   └── ml-monitoring.json
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboards.yaml
│           └── datasources/
│               └── prometheus.yaml
│
└── .github/
    └── workflows/
        ├── tests.yml
        ├── lint.yml
        └── docker.yml

```

### L'architecture repose sur des responsabilités bien distinctes :

```
┌──────────────────────────────────────────┐
│ src/traffic_prediction/                  │
│                                          │
│ LOGIQUE MÉTIER                           │
│ ingestion, features, ML, prediction      │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ shared/                                  │
│                                          │
│ CODE TRANSVERSAL                         │
│ config, logging, dates, paths            │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ dags/                                    │
│                                          │
│ ORCHESTRATION                            │
│ Airflow appelle le code de src/          │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ api/                                     │
│                                          │
│ SERVING                                  │
│ FastAPI expose les prédictions           │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ dashboard/                               │
│                                          │
│ VISUALISATION                            │
│ carte + métriques                        │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ docker/                                  │
│                                          │
│ BUILD DES IMAGES                         │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ deployments/                             │
│                                          │
│ CONFIGURATION INFRASTRUCTURE             │
│ nginx / prometheus / grafana             │
└──────────────────────────────────────────┘
```

### Le projet est découpé en 4 milestones.

Version	Objectif
- V0 — EDA	Télécharger trafic + référentiel, comprendre q, k, iu_ac, qualité des données
- V1 — ML baseline	Parquet → features temporelles → baseline → LightGBM → MLflow
- V2 — Enrichment	Ajouter météo + jours fériés + vacances + événements et mesurer leur impact
- V3 — Production	Airflow + FastAPI + PostgreSQL + dashboard + Docker + monitoring

