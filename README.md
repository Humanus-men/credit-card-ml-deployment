# Credit Card Default Prediction Service

Сервис машинного обучения для прогнозирования дефолта по кредитным картам.
Реализован полный цикл: от обучения модели до контейнеризации и A/B-тестирования.

Репозиторий: [https://github.com/Humanus-men/credit-card-ml-deployment](https://github.com/Humanus-men/credit-card-ml-deployment)

## Структура репозитория

```
credit-card-ml-deployment/
├── app/                    # Flask-приложение и обработчик модели
│   ├── __init__.py
│   ├── api.py              # Эндпоинты /predict и /health
│   └── model_handler.py    # Загрузка модели и предобработка
├── models/                 # Скрипт обучения и артефакты
│   ├── train_model.py      # Обучение двух версий модели
│   ├── model_v1.pkl        # Логистическая регрессия (после обучения)
│   ├── model_v2.pkl        # Случайный лес (после обучения)
│   ├── preprocessor.pkl    # ColumnTransformer (StandardScaler + OneHotEncoder)
│   └── feature_names.pkl   # Имена признаков после кодирования (справочно)
├── tests/                  # Автотесты
│   └── test_api.py
├── docker/                 # Docker-инфраструктура
│   └── Dockerfile
├── requirements.txt
├── docker-compose.yml      # Оркестрация
├── ab_test_plan.md         # План A/B-тестирования
├── ARCHITECTURE.md         # Архитектурные решения и MLOps-концепты
└── README.md
```

## Требования

- Python 3.10 или выше
- Docker (для контейнеризации)
- Файл датасета `UCI_Credit_Card.csv` (не включён в репозиторий; скачайте с https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset и поместите в корень проекта)

## Локальный запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/Humanus-men/credit-card-ml-deployment.git
cd credit-card-ml-deployment
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Обучение моделей
```bash
python models/train_model.py
```
После выполнения в папке `models/` появятся файлы:
- `model_v1.pkl` (логистическая регрессия)
- `model_v2.pkl` (случайный лес)
- `preprocessor.pkl` (ColumnTransformer)
- `feature_names.pkl` (имена признаков)

### 4. Запуск веб-сервиса
```bash
python app/api.py
```
Сервис запустится на `http://0.0.0.0:5000`.

### 5. Проверка здоровья
```bash
curl http://localhost:5000/health
```
Ответ: `{"status":"healthy"}`

### 6. Предсказание
#### Модель v1 (по умолчанию)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,
    "PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,
    "BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,
    "PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0
  }'
```

#### Модель v2 (для A/B-теста)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-Model-Version: v2" \
  -d '{
    "LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,
    "PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,
    "BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,
    "PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0
  }'
```

Формат ответа:
```json
{
  "prediction": 0 или 1,
  "probability": 0.23,
  "model_version": "v1"
}
```

## Запуск в Docker

### 1. Сборка образа
```bash
docker build -t credit-default-api -f docker/Dockerfile .
```

### 2. Запуск контейнера
```bash
docker run -p 5000:5000 credit-default-api
```

### 3. Использование готового образа
Образ доступен на Docker Hub:
```
docker pull gumanist39/credit-default-api:latest
docker run -p 5000:5000 gumanist39/credit-default-api:latest
```
Ссылка: [https://hub.docker.com/r/gumanist39/credit-default-api](https://hub.docker.com/r/gumanist39/credit-default-api)

## A/B-тестирование

Сервис поддерживает две версии модели: **v1** (Logistic Regression) и **v2** (Random Forest).  
Переключение осуществляется через HTTP-заголовок `X-Model-Version`.

Подробный план A/B-теста, метрики и статистический анализ описаны в файле [`ab_test_plan.md`](ab_test_plan.md).

## Документация

- Архитектурные решения и описание MLOps-концептов (RabbitMQ, логирование, DVC, MLflow, бизнес-метрики) – [`ARCHITECTURE.md`](ARCHITECTURE.md)
- План A/B-теста – [`ab_test_plan.md`](ab_test_plan.md)

## Тестирование

Запуск unit-тестов:
```bash
pytest tests/
```

## Особенности реализации

- **Предобработка**: категориальные признаки `SEX`, `EDUCATION`, `MARRIAGE` кодируются One-Hot Encoding; числовые признаки стандартизируются (`StandardScaler`). Пайплайн сохранён в `preprocessor.pkl`.
- **Балансировка классов**: для борьбы с дисбалансом дефолтов модели обучаются с `class_weight='balanced'`, что повышает полноту (Recall) для класса «дефолт».
- **Логирование**: запросы и ответы пишутся в stdout в формате JSON.
- **Версионирование моделей**: две версии загружаются одновременно, что позволяет проводить A/B-тестирование без перезапуска сервиса.