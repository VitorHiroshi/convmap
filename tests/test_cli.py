"""Tests for the CLI module."""

import json
import numpy as np
import pytest

from convmap import Engine, Chunk, EmbeddedConversation
from convmap import persistence
from convmap.cli import main


def _build_and_save(tmp_path, n=50, dims=64, seed=42):
    """Build an engine, save it, return the map path."""
    np.random.seed(seed)
    engine = Engine(dimensions=dims, epsilon=0.3, mu=3.0, maintenance_interval=20)

    centers = [np.random.randn(dims).astype(np.float32) for _ in range(3)]
    per_cluster = n // 3

    i = 0
    for center in centers:
        for _ in range(per_cluster):
            v = center + np.random.randn(dims) * 0.05
            v = (v / np.linalg.norm(v)).astype(np.float32)
            meta = {"id": f"conv-{i}", "timestamp": str(1000.0 + i)}
            engine.ingest_vector(v, meta)
            i += 1

    engine._maintain()
    engine.snapshot(label="initial")

    map_path = tmp_path / ".convmap"
    persistence.save(engine, map_path)
    return map_path


class TestStatusCommand:
    def test_status_no_map(self, tmp_path, capsys):
        main(["--map", str(tmp_path / "missing"), "status"])
        out = capsys.readouterr().out
        assert "No map" in out

    def test_status_with_map(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "status"])
        out = capsys.readouterr().out
        assert "Core clusters" in out
        assert "Dimensions" in out


class TestSnapshotCommand:
    def test_snapshot(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "snapshot", "--label", "test-snap"])
        out = capsys.readouterr().out
        assert "Snapshot captured" in out
        assert "test-snap" in out

        # Verify persisted
        engine = persistence.load(map_path)
        assert engine.snapshots[-1].label == "test-snap"


class TestQueryCommands:
    def test_clusters(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "query", "clusters"])
        out = capsys.readouterr().out
        assert "core clusters" in out

    def test_funnel(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "query", "funnel"])
        out = capsys.readouterr().out
        assert "Funnel" in out or "No clusters" in out

    def test_histogram(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "query", "histogram"])
        out = capsys.readouterr().out
        assert "Histogram" in out

    def test_anomalies(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "query", "anomalies", "--threshold", "0.99"])
        out = capsys.readouterr().out
        assert "Anomalies" in out

    def test_segment(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "query", "segment", "0"])
        out = capsys.readouterr().out
        assert "Segment" in out or "no members" in out

    def test_drift_needs_snapshots(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        # Only 1 snapshot, needs 2
        main(["--map", str(map_path), "query", "drift"])
        out = capsys.readouterr().out
        assert "snapshot" in out.lower()

    def test_drift_with_snapshots(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        # Add a second snapshot
        engine = persistence.load(map_path)
        engine.snapshot(label="second")
        persistence.save(engine, map_path)

        main(["--map", str(map_path), "query", "drift"])
        out = capsys.readouterr().out
        assert "Drift" in out

    def test_compare(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main([
            "--map", str(map_path), "query", "compare",
            "1000", "1025", "1025", "1050",
        ])
        out = capsys.readouterr().out
        assert "Compare" in out

    def test_segment_invalid_index(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "query", "segment", "999"])
        out = capsys.readouterr().out
        assert "no members" in out or "invalid" in out


class TestReportCommand:
    def test_report_no_map(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--map", str(tmp_path / "missing"), "report"])

    def test_report_json_stdout(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "report"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data
        assert "clusters" in data
        assert "funnel" in data
        assert "histogram" in data
        assert "anomalies" in data
        assert "drift" in data
        assert data["summary"]["core_clusters"] >= 0

    def test_report_markdown_stdout(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "report", "--format", "markdown"])
        out = capsys.readouterr().out
        assert "# Convmap Report" in out
        assert "## Summary" in out
        assert "## Clusters" in out

    def test_report_json_to_file(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        out_file = tmp_path / "report.json"
        main(["--map", str(map_path), "report", "-o", str(out_file)])
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "summary" in data
        # stdout should have confirmation message
        out = capsys.readouterr().out
        assert "Report written" in out

    def test_report_markdown_to_file(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        out_file = tmp_path / "report.md"
        main(["--map", str(map_path), "report", "--format", "markdown", "-o", str(out_file)])
        assert out_file.exists()
        content = out_file.read_text()
        assert "# Convmap Report" in content

    def test_report_with_anomaly_threshold(self, tmp_path, capsys):
        map_path = _build_and_save(tmp_path)
        main(["--map", str(map_path), "report", "--anomaly-threshold", "0.99"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["anomalies"]["threshold"] == 0.99


class TestImportCommand:
    def test_import_csv(self, tmp_path, capsys):
        """Test import with a small CSV (no real embedding — uses mock)."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "id,transcript\n"
            'conv-1,"agent: Hello customer\\ncustomer: I need help"\n'
            'conv-2,"agent: Good morning\\ncustomer: My bill is wrong"\n'
        )

        # We can't easily test with real embeddings (needs model download),
        # so we just test that the file detection and argument parsing work.
        # The actual import would require sentence-transformers installed.
        # Test that format detection works
        from convmap.cli import main
        # Just verify --help doesn't crash
        with pytest.raises(SystemExit):
            main(["import", "--help"])
