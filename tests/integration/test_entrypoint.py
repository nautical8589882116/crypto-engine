"""Smoke test: entrypoint control plane serves health + status."""

import http.server
import threading
import time
from http.client import HTTPConnection

from quant_crypto.entrypoint import ControlHandler


def test_entrypoint_health():
    srv = http.server.HTTPServer(("127.0.0.1", 0), ControlHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    try:
        c = HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/health")
        r = c.getresponse()
        assert r.status == 200
        assert b'"ok"' in r.read()

        c.request("GET", "/api/v1/status")
        r = c.getresponse()
        body = r.read()
        assert r.status == 200
        assert b"killswitch" in body
        assert b"paper" in body
    finally:
        srv.shutdown()
