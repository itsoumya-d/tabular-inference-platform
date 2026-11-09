import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store


@pytest.fixture(autouse=True)
def _clean_store():
    store.clear()
    yield
    store.clear()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def csv_dataset(client):
    csv_bytes = b"age,city,score,joined\n30,NYC,10,2024-01-01\n40,LA,20,2024-02-01\n50,NYC,30,2024-03-01\n"
    r = client.post("/datasets/upload", files={"file": ("data.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def json_dataset(client):
    payload = b'[{"age": 30, "city": "NYC"}, {"age": 40, "city": "LA"}]'
    r = client.post("/datasets/upload", files={"file": ("data.json", payload, "application/json")})
    assert r.status_code == 200, r.text
    return r.json()
