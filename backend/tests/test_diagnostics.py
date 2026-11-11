import json

import pytest

from transformer_observatory.diagnostics import AnalysisDiagnostics


def test_nested_timings_record_failures_and_stop_monitor(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("transformer_observatory.diagnostics.time.monotonic", lambda: clock[0])
    path = tmp_path / "timings.json"
    diagnostics = AnalysisDiagnostics("fixture", path)
    diagnostics.start()
    with diagnostics.measure("forward"):
        clock[0] = 1
        with pytest.raises(ValueError), diagnostics.measure("manifest_save"):
            clock[0] = 3
            raise ValueError("injected failure")
        assert diagnostics.operation == "forward"
        clock[0] = 4
    diagnostics.close()
    assert not diagnostics.worker.is_alive()
    report = json.loads(path.read_text())
    assert report["timings"]["forward"]["total_seconds"] == 4
    assert report["timings"]["manifest_save"]["total_seconds"] == 2


def test_stall_records_stack_rate_limits_and_bounds_history(tmp_path, caplog):
    diagnostics = AnalysisDiagnostics("fixture", tmp_path / "timings.json")
    with diagnostics.measure("chunk_write"):
        activity = diagnostics.activity
        diagnostics._check_stall(activity + 9)
        assert diagnostics.stalls == []
        diagnostics._check_stall(activity + 11)
        assert len(diagnostics.stalls) == 1
        assert diagnostics.stalls[0]["operation"] == "chunk_write"
        assert "test_stall_records_stack" in diagnostics.stalls[0]["stack"]
        diagnostics._check_stall(activity + 12)
        assert len(diagnostics.stalls) == 1
        for i in range(25):
            diagnostics._check_stall(activity + 50 + i * 31)
        assert len(diagnostics.stalls) == 20
    diagnostics._check_stall(diagnostics.activity + 1)
    assert "no checkpoint" in caplog.text
    diagnostics.close()


def test_diagnostic_file_failure_does_not_fail_analysis(tmp_path, caplog):
    path = tmp_path / "not-a-directory"
    path.write_text("existing")
    diagnostics = AnalysisDiagnostics("fixture", path / "timings.json")
    diagnostics.close()
    assert "could not save timing diagnostics" in caplog.text
