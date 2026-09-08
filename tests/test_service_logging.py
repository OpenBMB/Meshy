"""Tests for per-service-process output logs."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_role_output_is_redirected_to_runtime_log(tmp_path):
    from meshy.service.runtime import RuntimeDir

    log_path = Path(RuntimeDir(str(tmp_path)).process_log_path("training-0", 2))
    assert log_path.name == "training-0-2.log"
    child = textwrap.dedent(
        f"""
        import sys
        from meshy.service.base import redirect_output

        redirect_output({str(log_path)!r})
        print("stdout from role")
        print("stderr from role", file=sys.stderr)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", child], check=True, capture_output=True
    )

    assert result.stdout == b""
    assert result.stderr == b""
    contents = log_path.read_text()
    assert "stdout from role" in contents
    assert "stderr from role" in contents
