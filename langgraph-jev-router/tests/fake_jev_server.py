"""A local HTTP server that speaks Jev's /v1/systemone wire format (typesafe-sdk 0.7.x).

Lets the test-suite exercise the *real* SDK client end to end without network or an API key.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeJevServer:
    def __init__(self, choice="billing", confidence=0.93, status=200):
        self.choice, self.confidence, self.status = choice, confidence, status
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if server.status != 200:
                    payload = json.dumps({"error": {"message": "boom"}}).encode()
                    self.send_response(server.status)
                else:
                    labels = list(body["questions"]["route"]["criteria"])
                    rest = (1 - server.confidence) / max(len(labels) - 1, 1)
                    payload = json.dumps({
                        "model": "jev-1.13.0",
                        "usage": {"input_tokens": 87, "output_tokens": 1},
                        "answers": {"route": {
                            "type": "choice",
                            "choice": server.choice,
                            "confidence": server.confidence,
                            "probabilities": {l: (server.confidence if l == server.choice else rest) for l in labels},
                        }},
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-typesafe-request-id", "req_test_123")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
