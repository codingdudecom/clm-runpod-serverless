import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handler import Worker, validate_input
from startup import encoder_command, probe_embedding, resolve_encoder, stop_children, wait_ready


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        self.worker = Worker(self.engine)
        self.payload = {"operation": "rank", "state": "Question", "candidates": ["A", "B"]}

    def test_rank_preserves_upstream_result(self):
        result = [{"rank": 1, "candidate": "A", "prob": 0.8}]
        self.engine.rank.return_value = result
        self.assertEqual(self.worker({"input": self.payload}), result)
        self.engine.rank.assert_called_once_with("Question", ["A", "B"], instructions=None, temperature=1.0)

    def test_all_typed_questions_are_forwarded(self):
        payload = json.loads(Path("examples/system_one.json").read_text())
        self.engine.answer.return_value = {"answers": {}, "usage": {}}
        self.assertEqual(self.worker(payload), self.engine.answer.return_value)
        self.engine.answer.assert_called_once_with(payload["input"]["state"], payload["input"]["questions"], temperature=1.0)

    def test_invalid_inputs_do_not_reach_engine(self):
        invalid = [None, {}, {"input": []}]
        for change in [{"operation": "chat"}, {"candidates": []}, {"candidates": [1]},
                       {"candidates": [" "]}, {"temperature": True}, {"temperature": 0},
                       {"temperature": math.nan}, {"temperature": math.inf}, {"temperature": 101},
                       {"state": None}, {"instructions": 4},
                       {"operation": "system_one", "questions": {}},
                       {"operation": "system_one", "questions": {"x": []}},
                       {"operation": "system_one", "questions": {"x": {"type": "choice", "criteria": []}}},
                       {"operation": "system_one", "questions": {"x": {"type": "score", "criteria": ["A"]}}},
                       {"operation": "system_one", "questions": {"x": {"type": "noul", "criteria": []}}}]:
            invalid.append({"input": dict(self.payload, **change)})
        for job in invalid:
            with self.subTest(job=job), self.assertRaises(ValueError):
                self.worker(job)
        self.engine.rank.assert_not_called()
        self.engine.answer.assert_not_called()

    def test_upstream_validation_error_marks_failure(self):
        self.engine.rank.side_effect = ValueError("malformed candidate")
        with self.assertRaisesRegex(ValueError, "Invalid request: malformed"):
            self.worker({"input": self.payload})

    def test_engine_failures_are_not_success_payloads(self):
        self.engine.rank.side_effect = RuntimeError("encoder unavailable")
        with self.assertLogs("handler", level="ERROR"), self.assertRaisesRegex(RuntimeError, "inspect worker logs"):
            self.worker({"input": self.payload})


class StartupTests(unittest.TestCase):
    def test_cache_uses_exact_revision_and_rejects_other_snapshot(self):
        versions = {"encoder_repo": "Qwen/Qwen3-8B", "encoder_revision": "pinned"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unrelated = root / "models--Qwen--Qwen3-8B/snapshots/other"
            unrelated.mkdir(parents=True)
            (unrelated / "config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Pinned encoder snapshot missing"):
                resolve_encoder(versions, root)
            snapshot = unrelated.with_name("pinned")
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "No safetensors"):
                resolve_encoder(versions, root)
            (snapshot / "model.safetensors").write_text("fixture")
            self.assertEqual(resolve_encoder(versions, root), snapshot)

    def test_encoder_is_private_and_last_pooled(self):
        with patch.dict(os.environ, {}, clear=True):
            command = encoder_command(Path("/model"))
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(json.loads(command[command.index("--pooler-config") + 1])["pooling_type"], "LAST")
        self.assertEqual(command[command.index("--max-model-len") + 1], "2048")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "8")

    def test_readiness_checks_dimension(self):
        import io
        for dimension in [4096, 1024]:
            body = json.dumps({"data": [{"embedding": [0.1] * dimension}]}).encode()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
                if dimension == 4096:
                    self.assertTrue(probe_embedding())
                else:
                    with self.assertRaisesRegex(RuntimeError, "4096"):
                        probe_embedding()

    def test_readiness_accepts_base64_and_rejects_nonfinite(self):
        import base64
        import io
        import struct
        for value in [0.1, math.nan]:
            encoded = base64.b64encode(struct.pack("<4096f", *([value] * 4096))).decode()
            body = json.dumps({"data": [{"embedding": encoded}]}).encode()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
                if math.isfinite(value):
                    self.assertTrue(probe_embedding())
                else:
                    with self.assertRaisesRegex(RuntimeError, "finite"):
                        probe_embedding()

    def test_encoder_death_is_detected(self):
        encoder = Mock(returncode=2)
        encoder.poll.return_value = 2
        with self.assertRaisesRegex(RuntimeError, "exited during startup"):
            wait_ready(encoder, 10, lambda: False)

    def test_shutdown_during_startup(self):
        with self.assertRaises(InterruptedError):
            wait_ready(Mock(), 10, lambda: True)

    def test_readiness_timeout(self):
        encoder = Mock()
        encoder.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            wait_ready(encoder, 0, lambda: False)

    @unittest.skipIf(os.name == "nt", "POSIX process groups")
    def test_real_child_receives_shutdown(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        try:
            stop_children([child])
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == "__main__":
    unittest.main()
