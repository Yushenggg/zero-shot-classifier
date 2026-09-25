"""Regression tests for the benchmark HTML report generator."""

from __future__ import annotations

import json

from benchmark import report


def _summary(name: str) -> dict:
    return {
        "dataset": {"examples": 3, "evaluated": 3, "errors": 0},
        "model": {"id": "test/model", "device": "cpu", "dtype": "float32"},
        "metrics": {},
        "timing": {"eval_seconds": 1.0},
    }


def test_discover_nested_dirs(tmp_path):
    out = tmp_path / "popular" / "mnist" / "out"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(_summary("mnist")))
    flat = tmp_path / "usps" / "out"
    flat.mkdir(parents=True)
    (flat / "summary.json").write_text(json.dumps(_summary("usps")))

    files = report.discover([str(tmp_path)])
    assert len(files) == 2
    assert {f.parent.parent.name for f in files} == {"mnist", "usps"}


def test_discover_accepts_explicit_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    path = out / "summary.json"
    path.write_text("{}")
    assert report.discover([str(path)]) == [path]


def test_discover_deduplicates(tmp_path):
    out = tmp_path / "mnist" / "out"
    out.mkdir(parents=True)
    path = out / "summary.json"
    path.write_text("{}")
    files = report.discover([str(path), str(tmp_path)])
    assert files == [path]


def test_load_reports_names_from_path(tmp_path):
    out = tmp_path / "popular" / "kmnist" / "out"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(_summary("kmnist")))
    reports = report.load_reports(report.discover([str(tmp_path)]))
    assert reports[0]["name"] == "kmnist"


def test_load_reports_handles_variant_run_dir(tmp_path):
    out = tmp_path / "popular" / "mnist" / "out-nocal"
    out.mkdir(parents=True)
    summary = _summary("mnist")
    summary["settings"] = {"calibrate": False}
    (out / "summary.json").write_text(json.dumps(summary))

    loaded = report.load_reports(report.discover([str(tmp_path)]))[0]
    assert loaded["name"] == "mnist"
    assert loaded["run_dir"] == "out-nocal"
    assert loaded["run"] == "uncalibrated"


def test_load_reports_labels_calibrated_run(tmp_path):
    out = tmp_path / "popular" / "mnist" / "out"
    out.mkdir(parents=True)
    summary = _summary("mnist")
    summary["settings"] = {"calibrate": True}
    (out / "summary.json").write_text(json.dumps(summary))
    loaded = report.load_reports(report.discover([str(tmp_path)]))[0]
    assert loaded["run"] == "calibrated"


def test_build_html_embeds_data_and_escapes(tmp_path):
    html = report.build_html([{"name": "</script>", "metrics": {}}])
    assert "__REPORT_DATA__" not in html
    assert "__GENERATED_AT__" not in html
    # The closing tag in the payload is escaped so it can't break out of <script>.
    assert "<\\/script>" in html


def test_main_writes_file(tmp_path, capsys):
    out = tmp_path / "popular" / "mnist" / "out"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(_summary("mnist")))

    target = tmp_path / "report.html"
    code = report.main([str(tmp_path), "-o", str(target)])
    assert code == 0
    assert target.exists()
    assert "mnist" in target.read_text()
    assert "wrote" in capsys.readouterr().out


def test_main_missing_returns_error(tmp_path, capsys):
    code = report.main([str(tmp_path / "nothing"), "-o", str(tmp_path / "r.html")])
    assert code == 1
    assert "no summary.json" in capsys.readouterr().err
