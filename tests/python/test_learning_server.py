"""Exercise local API, static assets, and resuming the same actual checks."""

import json
from pathlib import Path
import runpy
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from steps.serve import make_server


class LearningServerTests(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.start_server()

    def start_server(self, ppo_pointer=None):
        self.server = make_server(0, output=self.output.name, ppo_pointer=ppo_pointer)
        self.server.RequestHandlerClass.log_message = lambda *_args: None
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def tearDown(self):
        self.stop_server()
        self.output.cleanup()

    def get(self, path):
        with urlopen(self.base + path) as response:
            return json.load(response)

    def command(self, command):
        request = Request(self.base + "/api/command", json.dumps({"command": command}).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request) as response:
            return json.load(response)

    def test_page_assets_and_actual_input_are_served(self):
        for path in ("/steps/index.html", "/steps/learning.css", "/steps/learning.js"):
            with urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 100)
        source = self.get("/api/input")
        self.assertEqual(source["circuit"]["blocks"]["B1"], [5, 28])
        self.assertEqual(len(source["circuit"]["nets"]), 5)
        self.assertIn("inspect_action", source["code"]["src/environment/placement_env.py"])

    def test_ppo_records_are_read_without_changing_the_teaching_session(self):
        root = Path(__file__).resolve().parents[2]
        build = root / "steps/build"
        build.mkdir(parents=True, exist_ok=True)
        self.stop_server()
        with tempfile.TemporaryDirectory(dir=build) as folder:
            directory = Path(folder)
            pointer = directory / "latest.json"
            pointer.write_text(json.dumps({"directory": directory.relative_to(root).as_posix()}))
            (directory / "progress.json").write_text(json.dumps({"update": 3, "status": "completed"}))
            (directory / "replay-0.json").write_text(json.dumps({"steps": [{"action": [0, 0]}]}))
            (directory / "replay-0-update-3.json").write_text(json.dumps({"steps": [{"action": [1, 0]}]}))
            (directory / "teaching-update-0.json").write_text(json.dumps({"source_update": 0, "transitions": []}))
            (directory / "teaching.json").write_text(json.dumps({"source_update": 3, "transitions": [{"action": 1}]}))
            (directory / "rollout.json").write_text(json.dumps({"update": 3, "transitions": [{"action": 2}]}))
            (directory / "study.json").write_text(json.dumps({"before": [{"episodes": 32}], "after": [{"episodes": 32}]}))
            self.start_server(pointer)
            before = self.get("/api/session")
            self.assertEqual(self.get("/api/ppo")["progress"]["update"], 3)
            self.assertEqual(self.get("/api/ppo")["checkpoints"], [3])
            self.assertEqual(self.get("/api/ppo")["study"]["after"][0]["episodes"], 32)
            chosen = directory.relative_to(root).as_posix()
            old = self.get("/api/ppo?run=" + chosen + "&update=0")
            self.assertEqual(old["selected_update"], 0)
            self.assertEqual(old["teaching"]["source_update"], 0)
            self.assertIsNone(old["rollout"])
            self.assertEqual(self.get("/api/ppo")["teaching"]["source_update"], 3)
            with self.assertRaises(HTTPError):
                self.get("/api/ppo?run=../unregistered")
            self.assertEqual(self.get("/api/ppo/replay?index=0")["steps"][0]["action"], [0, 0])
            self.assertEqual(self.get("/api/ppo/replay?index=0&update=3")["steps"][0]["action"], [1, 0])
            with self.assertRaises(HTTPError):
                self.get("/api/ppo/replay?index=-1")
            self.assertEqual(before, self.get("/api/session"))

    def test_server_restart_restores_search_without_executing_selected_action(self):
        self.command("module")
        before = self.command("choose")
        events = self.get("/api/events")
        self.assertEqual(before["chosen"], [25, 0])
        self.assertNotIn("B1", before["observation"]["placements"])
        self.stop_server()
        self.start_server()
        self.assertEqual(self.get("/api/session"), before)
        self.assertEqual(self.get("/api/events"), events)
        saved = json.loads((Path(self.output.name) / "learning-session.json").read_text())
        self.assertEqual(saved["commands"], ["module", "choose"])
        self.assertTrue((Path(self.output.name) / "learning-events.json").exists())

    def test_rejected_requests_preserve_state_and_candidate_checks_are_queryable(self):
        self.command("module")
        state = self.command("choose")
        candidate = self.get("/api/attempt?module=1&index=24")
        self.assertFalse(candidate["report"]["legal"])
        for action in (lambda: self.command("invented"),
                       lambda: self.get("/api/attempt?module=-1&index=0"),
                       lambda: self.get("/api/attempt?module=1&index=9999")):
            with self.assertRaises(HTTPError) as context:
                action()
            self.assertEqual(context.exception.code, 400)
        self.assertEqual(self.get("/api/session"), state)
        reset = self.command("reset")
        self.assertEqual(reset["observation"]["placements"], {})
        self.assertEqual(reset["phase"], "observe")

    def test_launcher_reopens_existing_page_without_binding_a_second_server(self):
        script = Path(__file__).resolve().parents[2] / "steps/serve.py"
        argv = [str(script), "--open", "--port", str(self.server.server_port)]
        with patch.object(sys, "argv", argv), patch("webbrowser.open") as opened:
            with self.assertRaises(SystemExit) as context:
                runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(context.exception.code, 0)
            opened.assert_called_once_with(self.base + "/steps/reading.html")


if __name__ == "__main__":
    unittest.main()
