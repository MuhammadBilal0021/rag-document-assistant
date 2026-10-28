import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
@pytest.fixture(autouse=True)
def no_external_http(monkeypatch):
    import httpx
    import requests
    def blocked(*a,**k): raise AssertionError("Unexpected external HTTP request")
    monkeypatch.setattr(httpx.HTTPTransport,"handle_request",blocked)
    monkeypatch.setattr(requests.sessions.Session,"request",blocked)
