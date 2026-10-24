"""The markdown report built from metrics.csv."""

import csv

import pytest

from gcg_defense import io_utils, report

ROWS = [
    ("none", "no defence", "test", "colluding", "model_mi", 0.60),
    ("none", "no defence", "test", "colluding", "help_rate", 0.80),
    ("none", "no defence", "test", "independent", "util_kl", 0.0),
    ("none", "no defence", "test", "independent", "validity", 0.999),
    ("none", "no defence", "test", "independent", "entropy", 0.82),
    ("none", "no defence", "test", "independent", "best_response_gap", 0.18),
    ("none", "no defence", "test_sim", "state=none,n_rounds=50", "detection_rate", 1.0),
    ("none", "no defence", "test_sim", "state=none,n_rounds=50", "a_minus_c_win_rate", 0.29),
    ("none", "no defence", "test_sim", "state=none,n_rounds=50", "certificate_cp_resid", 0.39),
    ("gcg_suffix", "k=20", "test", "colluding", "model_mi", 0.05),
    ("gcg_suffix", "k=20", "test", "colluding", "help_rate", 0.36),
    ("gcg_suffix", "k=20", "test", "independent", "util_kl", 0.03),
    ("gcg_suffix", "k=20", "test", "independent", "validity", 0.99),
    ("gcg_suffix", "k=20", "test", "independent", "entropy", 0.85),
    ("gcg_suffix", "k=20", "test", "independent", "best_response_gap", 0.2),
    ("gcg_suffix", "k=20", "test_sim", "state=none,n_rounds=50", "detection_rate", 0.1),
    ("gcg_suffix", "k=20", "test_sim", "state=none,n_rounds=50", "a_minus_c_win_rate", 0.01),
    # an old-style label with no state prefix must not break the report
    ("gcg_suffix", "k=20", "test_sim", "n_rounds=50", "detection_rate", 0.1),
    ("gcg_suffix", "k=20", "test_transfer", "colluding", "model_mi", 0.4),
    ("gcg_suffix", "k=20", "test_transfer", "colluding", "help_rate", 0.5),
    ("gcg_suffix", "k=20", "test_transfer", "independent", "util_kl", 0.02),
]


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "metrics.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(io_utils.METRIC_FIELDS)
        for method, params, split, condition, metric, value in ROWS:
            model = "other/model" if split == "test_transfer" else "Qwen/Qwen2.5-1.5B-Instruct"
            writer.writerow([method, params, 0, model, split, condition, metric, value])
    return str(tmp_path)


def test_report_has_every_section(run_dir):
    text = report.build_report(run_dir)
    assert "## Headline metrics" in text
    assert "## Generalisation" in text
    assert "## Behavioural test (state = none)" in text
    assert "## Transfer" in text
    assert "CP_resid <= 0.390" in text


def test_report_orders_none_and_suffix_first(run_dir):
    text = report.build_report(run_dir)
    body = text.split("## Headline metrics")[1]
    assert body.index("`none`") < body.index("`gcg_suffix`")
    assert "0.6000" in body and "0.0500" in body


def test_report_tolerates_missing_values(tmp_path):
    path = tmp_path / "metrics.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(io_utils.METRIC_FIELDS)
        writer.writerow(
            ["none", "", 0, "m", "test", "colluding", "model_mi", 0.1]
        )
    text = report.build_report(str(tmp_path))
    assert "--" in text  # absent metrics render as a dash, not a crash


def test_write_report_creates_the_file(run_dir):
    path = report.write_report(run_dir)
    assert path.endswith("report.md")
    with open(path) as fh:
        assert "Headline metrics" in fh.read()


def test_no_metrics_file_is_handled(tmp_path):
    assert "no metrics" in report.build_report(str(tmp_path))
