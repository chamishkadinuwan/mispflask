import os
import pytest
import json
from main import app, run_prediction

# Ensure predictions are generated before tests
@pytest.fixture(scope="session", autouse=True)
def setup_data():
    run_prediction()
    os.makedirs("test_results", exist_ok=True)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


TEST_SUMMARY_PATH = os.path.join("test_results", "summary.txt")

def save_test_result(test_name, passed):
    with open(TEST_SUMMARY_PATH, "a") as f:
        f.write(f"{test_name}: {'PASS' if passed else 'FAIL'}\n")


def test_flood_predictions(client):
    rv = client.get('/api/flood-predictions')
    passed = rv.status_code == 200
    try:
        data = rv.get_json()
        passed = passed and isinstance(data, list)
    except Exception:
        passed = False
    save_test_result('test_flood_predictions', passed)
    assert passed


def test_resource_needs(client):
    rv = client.get('/api/resource-needs')
    passed = rv.status_code == 200
    try:
        data = rv.get_json()
        passed = passed and isinstance(data, list)
    except Exception:
        passed = False
    save_test_result('test_resource_needs', passed)
    assert passed


def test_resource_needs_monthly(client):
    rv = client.get('/api/resource-needs-monthly')
    passed = rv.status_code == 200
    try:
        data = rv.get_json()
        passed = passed and isinstance(data, list)
    except Exception:
        passed = False
    save_test_result('test_resource_needs_monthly', passed)
    assert passed


def test_flood_chart(client):
    rv = client.get('/api/flood-chart')
    passed = rv.status_code == 200 and rv.content_type == 'image/png'
    save_test_result('test_flood_chart', passed)
    assert passed


def test_validation_results(client):
    rv = client.get('/api/validation-results')
    passed = rv.status_code == 200
    try:
        data = rv.get_json()
        passed = passed and isinstance(data, list)
    except Exception:
        passed = False
    save_test_result('test_validation_results', passed)
    assert passed


def test_classification_metrics(client):
    rv = client.get('/api/classification-metrics')
    passed = rv.status_code in [200, 404]
    if rv.status_code == 200:
        try:
            data = rv.get_json()
            passed = passed and isinstance(data, list)
        except Exception:
            passed = False
    save_test_result('test_classification_metrics', passed)
    assert passed


def test_classification_metrics_image(client):
    rv = client.get('/api/classification-metrics-image')
    passed = rv.status_code == 200 and rv.content_type == 'image/png'
    save_test_result('test_classification_metrics_image', passed)
    assert passed


def test_regression_metrics(client):
    rv = client.get('/api/regression-metrics')
    passed = rv.status_code == 200 and rv.content_type == 'image/png'
    save_test_result('test_regression_metrics', passed)
    assert passed


def test_feature_importance(client):
    rv = client.get('/api/feature-importance')
    passed = rv.status_code == 200 and rv.content_type == 'image/png'
    save_test_result('test_feature_importance', passed)
    assert passed


def test_send_email(client):
    payload = {
        "City": "Test City",
        "Month": "2026-01",
        "Predicted Affected People": 100,
        "High_Impact_Flood_Probability": 0.75,
        "Water Liters": 300,
        "Food Packs": 200,
        "Medical Kits": 10,
        "Blankets": 100,
        "Sanitation Kits": 20
    }
    rv = client.post('/api/send-email',
                     data=json.dumps(payload),
                     content_type='application/json')
    passed = rv.status_code in [200, 500]
    try:
        data = rv.get_json()
        passed = passed and isinstance(data, dict)
    except Exception:
        passed = False
    save_test_result('test_send_email', passed)
    assert passed


@pytest.fixture(scope="session", autouse=True)
def generate_html_report(request):
    def create_html():
        results_dir = "test_results"
        html_path = os.path.join(results_dir, "test_report.html")
        summary_path = os.path.join(results_dir, "summary.txt")
        rows = []
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                for line in f:
                    test_name, status = line.strip().split(": ")
                    rows.append(f"<tr><td>{test_name}</td><td>{status}</td></tr>")
        html = f"""
        <html><head><title>Test Results</title></head><body>
        <h1>Test Results</h1>
        <table border='1'>
        <tr><th>Test Name</th><th>Status</th></tr>
        {''.join(rows)}
        </table>
        </body></html>
        """
        with open(html_path, "w") as f:
            f.write(html)
    request.addfinalizer(create_html)
