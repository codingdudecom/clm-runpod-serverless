import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from client import submit_and_wait


class ClientTests(unittest.TestCase):
    @patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key", "RUNPOD_ENDPOINT_ID": "endpoint"})
    def test_submit_once_and_poll_to_completion(self):
        completed = {"id": "job", "status": "COMPLETED", "output": [1]}
        with patch("client.request", side_effect=[{"id": "job", "status": "IN_QUEUE"},
                                                   {"status": "IN_PROGRESS"}, completed]) as request, \
                patch("client.time.sleep"):
            self.assertEqual(submit_and_wait({"input": {}}, interval=0), completed)
            self.assertEqual([call.args[0] for call in request.call_args_list], ["POST", "GET", "GET"])

    @patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key", "RUNPOD_ENDPOINT_ID": "endpoint"})
    def test_failed_job_raises(self):
        with patch("client.request", side_effect=[{"id": "job"}, {"status": "FAILED", "error": "bad input"}]), \
                patch("client.time.sleep"), self.assertRaisesRegex(RuntimeError, "bad input"):
            submit_and_wait({"input": {}}, interval=0)

    @patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key", "RUNPOD_ENDPOINT_ID": "endpoint"})
    def test_timeout_does_not_resubmit_or_cancel(self):
        with patch("client.request", return_value={"id": "job", "status": "IN_QUEUE"}) as request, \
                self.assertRaisesRegex(TimeoutError, "has not been cancelled"):
            submit_and_wait({"input": {}}, timeout=0)
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
