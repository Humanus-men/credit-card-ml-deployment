import pytest
from app.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

VALID_PAYLOAD = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
    "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}

def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'healthy'}

def test_predict_v1(client):
    resp = client.post('/predict', json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data and 'probability' in data
    assert data['model_version'] == 'v1'

def test_predict_v2(client):
    resp = client.post('/predict', json=VALID_PAYLOAD,
                       headers={'X-Model-Version': 'v2'})
    assert resp.status_code == 200
    assert resp.get_json()['model_version'] == 'v2'

def test_missing_features(client):
    incomplete = {"LIMIT_BAL": 10000}
    resp = client.post('/predict', json=incomplete)
    assert resp.status_code in [400, 500]  # ошибка из-за отсутствия признаков