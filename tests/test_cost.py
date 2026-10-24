"""The cost model and the sweep grid.

These guard the numbers I quote before asking for a launch: a wrong grid or a
wrong price is a real-money mistake, not a cosmetic one.
"""

import json
import os

import pytest

from gcg_defense.cost import Episode, Pricing, s_per_step_from_csv, speed_per_dollar, summarise
from gcg_defense.sweep import build_points, to_episodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICING = os.path.join(ROOT, "configs", "pricing.json")


@pytest.fixture(scope="module")
def pricing():
    return Pricing.load(PRICING)


def test_pricing_file_records_its_provenance(pricing):
    with open(PRICING) as fh:
        raw = json.load(fh)
    assert raw["_source"].startswith("https://modal.com")
    assert raw["_fetched"]
    assert pricing.gpu_per_second["A10G"] > 0
    assert pricing.cpu_per_core_second > 0


def test_every_gpu_we_offer_has_a_price(pricing):
    for gpu in ("T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100"):
        assert pricing.gpu_price(gpu) > 0


def test_unknown_gpu_is_refused_not_guessed(pricing):
    with pytest.raises(KeyError):
        pricing.gpu_price("B200")


def test_duration_and_cost_are_the_stated_formula(pricing):
    episode = Episode(
        name="x",
        gpu="A10G",
        seed=0,
        config="c",
        n_steps=100,
        eval_every=25,
        s_per_step=10.0,
        eval_seconds=5.0,
        startup_seconds=30.0,
        model_load_seconds=45.0,
        cpu_cores=1.0,
        memory_gib=8.0,
    )
    expected_seconds = 30 + 45 + 100 * 10 + 4 * 5
    assert episode.duration_seconds() == pytest.approx(expected_seconds)
    expected_cost = expected_seconds * (
        pricing.gpu_price("A10G")
        + pricing.cpu_per_core_second * 1.0
        + pricing.memory_per_gib_second * 8.0
    )
    assert episode.cost(pricing) == pytest.approx(expected_cost)


def test_missing_measurement_is_an_error_not_a_guess():
    episode = Episode(name="x", gpu="A10G", seed=0, config="c")
    with pytest.raises(ValueError, match="measured"):
        episode.duration_seconds()


def test_timeout_is_duration_times_safety_rounded_up():
    episode = Episode(name="x", gpu="A10G", seed=0, config="c", n_steps=10, s_per_step=10.0)
    assert episode.timeout_seconds(1.5) >= episode.duration_seconds() * 1.5
    assert episode.timeout_seconds() % 60 == 0


def test_speed_per_dollar_prefers_the_cheaper_card_when_it_is_fast_enough(pricing):
    # A10G is worth it when it is less than price_A100/price_A10G times slower.
    ratio = pricing.gpu_price("A100") / pricing.gpu_price("A10G")
    table = speed_per_dollar({"A100": 10.0, "A10G": 10.0 * (ratio - 0.1)}, pricing)
    assert table.splitlines()[2].strip().startswith("A10G")
    table = speed_per_dollar({"A100": 10.0, "A10G": 10.0 * (ratio + 0.5)}, pricing)
    assert table.splitlines()[2].strip().startswith("A100")


def test_sweep_grid_is_the_product_of_axes_and_seeds():
    points = build_points(
        seeds=[0, 1, 2], gpu="A10G", axis_values={"lambda_util": [0.1, 1.0, 10.0]}
    )
    assert len(points) == 9
    assert len({p.name for p in points}) == 9
    assert all("run.seed=" in p.override_string() for p in points)
    assert all("loss.lambda_util=" in p.override_string() for p in points)


def test_sweep_grid_without_axes_is_just_the_seeds():
    points = build_points(seeds=[0, 1], gpu="L4")
    assert len(points) == 2
    assert all(p.tag == "default" for p in points)
    assert all(p.gpu == "L4" for p in points)


def test_unknown_axis_is_refused():
    with pytest.raises(ValueError, match="unknown sweep axis"):
        build_points(axis_values={"learning_rate": [1.0]})


def test_episodes_refuse_to_cost_an_unmeasured_gpu():
    points = build_points(seeds=[0], gpu="A10G")
    with pytest.raises(ValueError, match="no measured s/step"):
        to_episodes(points, s_per_step={"A100": 6.4})


def test_summary_totals_match_the_sum_of_episodes(pricing):
    points = build_points(seeds=[0, 1], gpu="A10G", axis_values={"lambda_util": [1.0]})
    episodes = to_episodes(points, s_per_step={"A10G": 12.0}, n_steps=50)
    text = summarise(episodes, pricing, max_concurrent=1)
    total = sum(e.cost(pricing) for e in episodes)
    assert f"${total:.2f}" in text
    assert "upper bound" in text
    assert pricing.source in text


def test_s_per_step_from_csv_skips_the_first_step(tmp_path):
    path = tmp_path / "steps.csv"
    path.write_text("step,seconds\n0,99.0\n1,10.0\n2,12.0\n3,10.0\n")
    assert s_per_step_from_csv(str(path)) == pytest.approx(10.0)


def test_s_per_step_from_csv_rejects_an_empty_file(tmp_path):
    path = tmp_path / "steps.csv"
    path.write_text("step,other\n0,1\n")
    with pytest.raises(ValueError):
        s_per_step_from_csv(str(path))
