"""Serve the maintained learning page and run its checks in the Python environment."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import parse_qs, urlparse
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.visualization.learning_session import LearningSession
from integration.visualization.ppo_data import read_run
from src.data.parse_blknet import load_circuit


def make_server(port=8868, input_path=None, output=None, ppo_pointer=None):
    source = Path(input_path) if input_path else ROOT / "benchmarks/alarik-synthetic/blknet_5.txt"
    circuit = load_circuit(source)
    output = Path(output) if output else ROOT / "steps/build"
    ppo_pointer = Path(ppo_pointer) if ppo_pointer else ROOT / "steps/build/ppo-latest.json"
    if not ppo_pointer.exists():
        ppo_pointer = ROOT / "steps/evidence/ppo-latest.json"
    output.mkdir(parents=True, exist_ok=True)
    saved_path = output / "learning-session.json"
    session = LearningSession(circuit)
    canonical = json.loads(json.dumps(circuit))
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved.get("circuit") == canonical:
            for command in saved["commands"]:
                session.command(command)
    lock = Lock()

    def save():
        temporary = output / "learning-session.pending.json"
        temporary.write_text(json.dumps({"circuit": circuit, "commands": session.commands}, separators=(",", ":")), encoding="utf-8")
        temporary.replace(saved_path)
        (output / "learning-events.json").write_text(json.dumps(session.events, separators=(",", ":")), encoding="utf-8")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def json_response(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            path = urlparse(self.path)
            if path.path == "/":
                self.send_response(302)
                self.send_header("Location", "/steps/reading.html")
                self.end_headers()
            elif path.path == "/api/session":
                with lock:
                    self.json_response(session.snapshot())
            elif path.path == "/api/input":
                self.json_response({"circuit": circuit, "raw": source.read_text(encoding="utf-8"),
                                    "source": source.name,
                                    "code": {name: (ROOT / name).read_text(encoding="utf-8") for name in (
                                        "src/data/parse_blknet.py", "src/environment/placement_env.py",
                                        "integration/visualization/learning_session.py")}})
            elif path.path == "/api/attempt":
                try:
                    query = parse_qs(path.query)
                    module, index = int(query["module"][0]), int(query["index"][0])
                    if min(module, index) < 0:
                        raise ValueError("Negative indices are not allowed")
                    with lock:
                        self.json_response(session.attempt(module, index))
                except (ValueError, KeyError, IndexError) as error:
                    self.json_response({"error": str(error)}, 400)
            elif path.path == "/api/events":
                with lock:
                    self.json_response(session.events)
            elif path.path == "/api/comparison":
                pointer = ROOT / "steps/build/comparison-latest.json"
                if not pointer.exists():
                    pointer = ROOT / "steps/evidence/comparison-latest.json"
                if pointer.exists():
                    directory = (ROOT / json.loads(pointer.read_text(encoding="utf-8"))["directory"]).resolve()
                    if directory.is_relative_to(ROOT):
                        self.json_response(json.loads((directory / "summary.json").read_text(encoding="utf-8")))
                    else:
                        self.json_response({"error": "Comparison path must stay inside the repository"}, 400)
                else:
                    self.json_response({"status": "unavailable", "cases": []})
            elif path.path in ("/api/ppo", "/api/ppo/replay"):
                try:
                    index = int(parse_qs(path.query).get("index", ["0"])[0]) if path.path.endswith("replay") else None
                    update_text = parse_qs(path.query).get("update", [None])[0]
                    update = int(update_text) if update_text is not None else None
                    run = parse_qs(path.query).get("run", [None])[0]
                    self.json_response(read_run(ROOT, ppo_pointer, index, update, run))
                except (ValueError, OSError, KeyError) as error:
                    self.json_response({"error": str(error)}, 400)
            else:
                super().do_GET()

        def do_POST(self):
            nonlocal session
            if self.path != "/api/command":
                self.json_response({"error": "Unknown endpoint"}, 404)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                with lock:
                    if data.get("command") == "reset":
                        session = LearningSession(circuit)
                        snapshot = session.snapshot()
                    else:
                        snapshot = session.command(data.get("command"))
                    save()
                    self.json_response(snapshot)
            except (ValueError, TypeError) as error:
                self.json_response({"error": str(error)}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8868)
    parser.add_argument("--open", action="store_true", help="Open the maintained page in the default browser")
    args = parser.parse_args()
    if args.open:
        import webbrowser
        try:
            with urlopen(f"http://127.0.0.1:{args.port}/api/input", timeout=1) as response:
                existing = json.load(response)
            if existing.get("code", {}).get("src/environment/placement_env.py") == (ROOT / "src/environment/placement_env.py").read_text(encoding="utf-8"):
                webbrowser.open(f"http://127.0.0.1:{args.port}/steps/reading.html")
                raise SystemExit(0)
        except (URLError, OSError, ValueError):
            pass
    server = make_server(args.port)
    print(f"Learning page: http://127.0.0.1:{server.server_port}/steps/reading.html", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}/steps/reading.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
