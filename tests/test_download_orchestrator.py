#!/usr/bin/env python3
"""
Unit tests for AVA download orchestrator reliability paths.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_orchestrator import AVADownloadOrchestrator


class TestAnnotationValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video_dir = root / "videos"
        self.ann_dir = root / "annotations"
        self.orch = AVADownloadOrchestrator(
            video_dir=str(self.video_dir),
            annotation_dir=str(self.ann_dir),
            max_workers=1
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_validator_rejects_html_payload(self):
        path = self.ann_dir / "ava_train_v2.2.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "<!DOCTYPE html><html><title>Error 404</title></html>",
            encoding="utf-8"
        )
        self.assertFalse(self.orch._validate_annotation_file(path, path.name))

    def test_validator_accepts_ava_train_shape(self):
        path = self.ann_dir / "ava_train_v2.2.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "1j20qq1JyX4,0902,0.1,12,0.1,0.1,0.2,0.2\n",
            encoding="utf-8"
        )
        self.assertTrue(self.orch._validate_annotation_file(path, path.name))


class TestAnnotationFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video_dir = root / "videos"
        self.ann_dir = root / "annotations"
        self.orch = AVADownloadOrchestrator(
            video_dir=str(self.video_dir),
            annotation_dir=str(self.ann_dir),
            max_workers=1
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_existing_file_is_replaced_via_fallback(self):
        filename = "ava_train_v2.2.csv"
        dest = self.ann_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("<html>404</html>", encoding="utf-8")

        calls = []

        def fake_download(url, output, timeout=600):
            calls.append(url)
            if url == "https://mirror-1/file.csv":
                return False
            output.write_text("1j20qq1JyX4,0902,0.1,12,0.1,0.1,0.2,0.2\n", encoding="utf-8")
            return True

        with patch.object(self.orch, "_build_annotation_candidates", return_value=[
            "https://mirror-1/file.csv",
            "https://mirror-2/file.csv",
        ]), patch.object(self.orch, "_download_file", side_effect=fake_download):
            ok = self.orch._download_and_validate_annotation(filename, dest)

        self.assertTrue(ok)
        self.assertEqual(calls, ["https://mirror-1/file.csv", "https://mirror-2/file.csv"])
        self.assertNotIn("<html", dest.read_text(encoding="utf-8").lower())

    def test_failed_first_mirror_then_success_second(self):
        filename = "ava_file_names_trainval_v2.1.txt"
        dest = self.ann_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        calls = []

        def fake_download(url, output, timeout=600):
            calls.append(url)
            if url == "https://m1/list.txt":
                return False
            output.write_text("1j20qq1JyX4.mp4\n", encoding="utf-8")
            return True

        with patch.object(self.orch, "_build_annotation_candidates", return_value=[
            "https://m1/list.txt",
            "https://m2/list.txt",
        ]), patch.object(self.orch, "_download_file", side_effect=fake_download):
            ok = self.orch._download_and_validate_annotation(filename, dest)

        self.assertTrue(ok)
        self.assertEqual(calls, ["https://m1/list.txt", "https://m2/list.txt"])


class TestProposalPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video_dir = root / "videos"
        self.ann_dir = root / "annotations"
        self.orch = AVADownloadOrchestrator(
            video_dir=str(self.video_dir),
            annotation_dir=str(self.ann_dir),
            max_workers=1
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_proposal_download_failures_warn_and_continue(self):
        with patch.object(self.orch, "_download_file", return_value=False):
            with self.assertLogs("scripts.download_orchestrator", level="WARNING") as cm:
                self.orch._download_proposals()

        logs = "\n".join(cm.output)
        self.assertIn("Optional proposal file unavailable", logs)


class TestStrictDownloadFailureHandling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video_dir = root / "videos"
        self.ann_dir = root / "annotations"
        self.orch = AVADownloadOrchestrator(
            video_dir=str(self.video_dir),
            annotation_dir=str(self.ann_dir),
            max_workers=1
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_file_returns_false_on_http_error(self):
        output = self.video_dir / "x.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        cp = subprocess.CompletedProcess(
            args=["curl"],
            returncode=22,
            stdout="",
            stderr="curl: (22) The requested URL returned error: 404",
        )

        with patch("scripts.download_orchestrator.shutil.which", return_value=None):
            with patch("scripts.download_orchestrator.subprocess.run", return_value=cp):
                with self.assertLogs("scripts.download_orchestrator", level="WARNING") as cm:
                    ok = self.orch._download_file("https://example.com/missing", output, timeout=2)

        self.assertFalse(ok)
        self.assertFalse(output.exists())
        self.assertIn("HTTP error while downloading", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
