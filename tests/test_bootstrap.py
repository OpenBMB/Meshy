"""Bootstrap store + readiness markers: the file-less startup sync plane.

The readiness markers used to be empty files under ``<runtime>/ready/``; they
now live in the run's bootstrap TCPStore (``meshy/service/bootstrap.py``). These
tests cover the marker API in-process, marker visibility across a real process
boundary (the deployed shape: a child publishes, the parent's ``wait_until``
gate consumes), and the per-run key namespacing that keeps back-to-back runs in
one process (pytest) from seeing each other's markers.

Run with the torchtitan env:

    TRANSFER_QUEUE_SRC=/workspace/TransferQueue \\
    python -m pytest tests/test_bootstrap.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_marker_roundtrip_and_timeout(tmp_path):
    from meshy.service.runtime import RuntimeDir

    rt = RuntimeDir(str(tmp_path / "run"))
    assert not rt.is_ready("inference-0")
    rt.mark_ready("inference-0")
    assert rt.is_ready("inference-0")
    rt.wait_ready(["inference-0"], timeout=5)

    with pytest.raises(TimeoutError, match=r"training-0"):
        rt.wait_ready(["inference-0", "training-0"], timeout=1.0, interval=0.1)


def test_marker_namespaced_per_run(tmp_path):
    """The same service name in two runs must not share a marker."""
    from meshy.service.runtime import RuntimeDir

    a = RuntimeDir(str(tmp_path / "run_a"))
    b = RuntimeDir(str(tmp_path / "run_b"))
    a.mark_ready("inference-0")
    assert a.is_ready("inference-0")
    assert not b.is_ready("inference-0")


def test_marker_crosses_process_boundary(tmp_path):
    """A child process publishes; the parent's readiness gate observes it.

    This is the deployed arrangement (modulo direction): processes share only
    the inherited ``XRL_BOOTSTRAP_ADDR``, no filesystem paths.
    """
    from meshy.service import bootstrap
    from meshy.service.runtime import RuntimeDir

    root = str(tmp_path / "run")
    rt = RuntimeDir(root)
    # Resolve the store first so XRL_BOOTSTRAP_ADDR is exported for the child.
    bootstrap.get_store()

    child = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REPO_ROOT!r})
        from meshy.service.runtime import RuntimeDir
        RuntimeDir({root!r}).mark_ready("training-0")
    """)
    subprocess.run([sys.executable, "-c", child], check=True, timeout=60)
    rt.wait_ready(["training-0"], timeout=10)


def test_wait_keys_liveness_probe_fails_fast(tmp_path):
    from meshy.service import bootstrap

    class Boom(RuntimeError):
        pass

    def liveness():
        raise Boom("publisher died")

    with pytest.raises(Boom):
        bootstrap.wait_keys(
            [f"never|{tmp_path}"], timeout=30, interval=0.05, liveness=liveness
        )
