"""TensorBoard tag hygiene: a tag is either a scalar or a histogram.

Writing both under one name (as the old metrics adapter did for
``rollout/advantages`` & co.) makes the scalars plugin answer HTTP 500 for
that tag.
"""

from __future__ import annotations

import pytest

from meshy.backend.titan.metrics import HISTOGRAM_PREFIX, _TensorBoard


def test_tensorboard_histograms_do_not_share_scalar_tags(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing import plugin_event_multiplexer
    from tensorboard.backend.event_processing import data_provider
    from tensorboard.data import provider
    from tensorboard import context

    board = _TensorBoard(str(tmp_path), enabled=True)
    metrics = {"rollout/advantages": 0.0, "rollout/log_probs": -0.5, "rollout/rollout_log_probs": -0.5}
    hist = {"rollout/advantages": [0.1, -0.1], "rollout/log_probs": [-0.4, -0.6],
            "rollout/rollout_log_probs": [-0.4, float("nan")]}
    for step in (1, 2):
        board.log(metrics, step, hist)
    board.close()

    mux = plugin_event_multiplexer.EventMultiplexer()
    mux.AddRunsFromDirectory(str(tmp_path))
    mux.Reload()
    dp = data_provider.MultiplexerDataProvider(mux, str(tmp_path))
    ctx = context.RequestContext()
    (run,) = mux.Runs().keys()
    scalars = dp.list_scalars(ctx, experiment_id="", plugin_name="scalars")[run]
    histograms = dp.list_tensors(ctx, experiment_id="", plugin_name="histograms")[run]
    assert set(scalars) == set(metrics)
    assert set(histograms) == {HISTOGRAM_PREFIX + k for k in hist}
    # Reading every scalar tag succeeds (this raised for the dual-logged tags).
    for tag in metrics:
        read = dp.read_scalars(
            ctx, experiment_id="", plugin_name="scalars", downsample=100,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        assert [d.step for d in read[run][tag]] == [1, 2]


def test_tensorboard_skips_non_finite_scalars(tmp_path):
    pytest.importorskip("tensorboard")
    board = _TensorBoard(str(tmp_path), enabled=True)
    board.log({"a": float("nan"), "b": 1.0}, 1)
    board.close()
    from tensorboard.backend.event_processing import plugin_event_multiplexer

    mux = plugin_event_multiplexer.EventMultiplexer()
    mux.AddRunsFromDirectory(str(tmp_path))
    mux.Reload()
    (run,) = mux.Runs().keys()
    assert set(mux.PluginRunToTagToContent("scalars")[run]) == {"b"}
