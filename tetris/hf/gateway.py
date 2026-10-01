"""One local decision API over Ollaya and the authors' native Python servers."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# No substituted models: each route is an explicit native runtime and wire alias.
ROUTES = {
    "winnow:e4b": ("http://127.0.0.1:11435", "winnow:e4b"),
    "laya:en-mps": ("http://127.0.0.1:11436", "laya:en-mps"),
    "kev:4b-mps": ("http://127.0.0.1:11437", "kev-latest"),
    "julia-1:mps": ("http://127.0.0.1:11438", "julia-1:mps"),
    "kev:4b-mlx": ("http://127.0.0.1:11439", "kev-latest"),
    "laya:multilingual-mps": ("http://127.0.0.1:11441", "laya:multilingual-mps"),
    "lev:4b-mps": ("http://127.0.0.1:11442", "lev:4b-mps"),
    "clm:8b-mps": ("http://127.0.0.1:11443", "clm:8b-mps"),
}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body):
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/v1/models":
            return self.send(404, {"error": "Unknown path"})
        cards = []
        for model, (endpoint, wire) in ROUTES.items():
            try:
                with urlopen(endpoint + "/v1/models", timeout=2) as r:
                    native = json.load(r)
                cards.append({"name": model, "native_model": wire, "endpoint": endpoint,
                              "available": True, "native": native})
            except (URLError, OSError):
                cards.append({"name": model, "available": False})
        self.send(200, {"models": cards})

    def do_POST(self):
        if self.path not in ("/v1/systemone", "/v1/decisions"):
            return self.send(404, {"error": "Unknown path"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                return self.send(413, {"error": "Request limit is 64 KiB"})
            body = json.loads(self.rfile.read(length))
            model = body["model"]
            endpoint, wire = ROUTES[model]
            native_body = {**body, "model": wire}
            request = Request(endpoint + "/v1/systemone", data=json.dumps(native_body).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=300) as r:
                result = json.load(r)
            result["gateway"] = {"native_model": result.get("model"), "native_endpoint": endpoint}
            result["model"] = model
            self.send(200, result)
        except HTTPError as exc:
            self.send(exc.code, {"native_error": exc.read().decode()[:4000]})
        except KeyError:
            self.send(404, {"error": "Unknown or missing model"})
        except (ValueError, TypeError) as exc:
            self.send(422, {"error": str(exc)})
        except (URLError, OSError) as exc:
            self.send(503, {"error": str(exc)})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 11440), Handler).serve_forever()
