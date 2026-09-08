from __future__ import annotations


def test_runtime_artifact_dirs_follow_explicit_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("XRL_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("XRL_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("XRL_TENSORBOARD_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("XRL_TRAJECTORY_DIR", str(tmp_path / "trajectories"))

    from meshy.service.runtime import RuntimeDir

    runtime = RuntimeDir(str(tmp_path / "runtime"))

    assert runtime.checkpoint_path("training-0", 3) == str(
        tmp_path / "checkpoints" / "training-0" / "v3"
    )
    assert runtime.process_log_path("training-0", 1) == str(
        tmp_path / "logs" / "training-0-1.log"
    )
    assert runtime.tensorboard_path("training-0") == str(
        tmp_path / "events" / "training-0"
    )
    assert runtime.trajectory_path() == str(
        tmp_path / "trajectories" / "trajectories.jsonl"
    )


def test_runtime_artifact_dirs_keep_existing_defaults(tmp_path, monkeypatch):
    for name in (
        "XRL_CHECKPOINT_DIR",
        "XRL_LOG_DIR",
        "XRL_TENSORBOARD_DIR",
        "XRL_TRAJECTORY_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    from meshy.service.runtime import RuntimeDir

    runtime = RuntimeDir(str(tmp_path / "runtime"))

    assert runtime.checkpoint_dir == str(tmp_path / "runtime" / "weights")
    assert runtime.log_dir == str(tmp_path / "runtime" / "logs")
    assert runtime.tensorboard_dir == str(tmp_path / "runtime" / "tensorboard")
    assert runtime.trajectory_dir == str(tmp_path / "runtime")
    assert runtime.trajectory_path() == str(tmp_path / "runtime" / "trajectories.jsonl")
