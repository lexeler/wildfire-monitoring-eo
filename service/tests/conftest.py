import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from service.app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def cat():
    from service.app.main import get_catalog

    return get_catalog()
