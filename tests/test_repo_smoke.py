"""Smoke tests on the repository itself: notebooks parse, saved outputs are present, files referenced exist,
pins are pins, and no course-style wording is in the text."""
import json
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

RESULT_FILES_05 = [
    "ex1_pod_modes.csv", "ex1_pod_cumulative_variance.csv", "ex1_pod_modes.png", "ex2_pod_vs_vae_runs.csv",
    "ex2_pod_vs_vae_summary.csv", "ex3_error_by_region.csv", "ex3_error_maps.png", "ex4_error_split_runs.csv",
    "ex4_error_split_summary.csv", "ex5_aerodynamic_head_runs.csv", "ex5_aerodynamic_head_summary.csv",
    "ex6_design_search.csv", "ex6_best_design_pressure.png", "ex7_polars.csv", "ex7_polars.png",
    "ex8_distance_to_training.csv", "ex9_ensemble_grid.csv", "ex9_calibration_test.csv", "ex9_ensemble_spread.png",
    "ex9_calibration.png", "ex10_latent_statistics.csv", "ex10_latent_correlations.png", "ex10_latent_traversal.png",
    "ex11_query_cost.csv",
]
RESULT_FILES_06 = [
    "A_polars_at_10_4_operating_point.csv", "A_summary.csv", "A_polars.png", "B_timing.csv", "C_cross_test_runs.csv",
    "C_cross_test_summary.csv", "D_flight_moments.csv", "D_pressure_fields.png", "E_training_ranges.csv",
    "E_sweep_outside.csv", "E_probe_18deg.csv", "E_outside_range.png",
]
COURSE_WORDS = re.compile(r"\b(course|cohort|instructor|lesson|lecture|homework|bootcamp|professor|syllabus|week \d+|module \d+)\b", re.I)


@pytest.mark.parametrize("stage", ["05", "06"])
def test_notebook_is_valid_and_fully_executed(repo, stage, notebooks):
    path = next(repo.glob(f"notebooks/{stage}_*.ipynb"))
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code" and "".join(c.source).strip()]
    assert code_cells, "no code cells"
    assert all(c.execution_count is not None for c in code_cells), "a code cell was never executed"
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs), "a cell errored"
    assert path.with_suffix(".html").exists(), "HTML export missing"


def test_results_and_models_referenced_by_the_notebooks_exist(repo):
    for name in RESULT_FILES_05:
        assert (repo / "results/05_airfrans_rom_betavae_design" / name).exists(), name
    for name in RESULT_FILES_06:
        assert (repo / "results/06_field_model_in_control_loop" / name).exists(), name
    m05 = repo / "models/05_airfrans_rom_betavae_design"
    for name in ["pod_pca_8.joblib", "beta_vae_encoder.keras", "beta_vae_decoder.keras", "parameter_scaler.joblib",
                 "chain_scalers.joblib", "channel_statistics.npz"]:
        assert (m05 / name).exists(), name
    for kind in ("pod", "vae"):
        for seed, folder in ((2026, m05), (2027, m05 / "exercises"), (2028, m05 / "exercises")):
            for stem in (f"latent_surrogate_{kind}_seed{seed}.keras", f"latent_surrogate_{kind}_seed{seed}.json",
                         f"aerodynamic_head_{kind}_seed{seed}.keras", f"aerodynamic_head_{kind}_seed{seed}_latent_scaler.joblib"):
                assert (folder / stem).exists(), stem
    m06 = repo / "models/06_field_model_in_control_loop"
    agents = sorted(p.name for p in m06.glob("*.zip"))
    assert agents == ["ppo_l12chain_seed0.zip", "ppo_l12chain_seed1.zip", "ppo_l12chain_seed2.zip",
                      "ppo_surrogate_seed1.zip", "ppo_surrogate_seed2.zip"]       # the sixth agent comes from the companion repository
    for p in m06.glob("*.zip"):
        assert p.with_name(p.stem + "_monitor.csv").exists()
    for name in ["airfrans_4digit_coefficients.csv", "airfrans_cl_cd_surrogate.keras", "airfrans_cl_cd_scalers.joblib"]:
        assert (repo / "data" / name).exists()


def test_surrogate_metadata_is_consistent_with_query_cost_table(repo):
    import pandas as pd
    cost = pd.read_csv(repo / "results/05_airfrans_rom_betavae_design/ex11_query_cost.csv")
    m05 = repo / "models/05_airfrans_rom_betavae_design"
    for meta in list(m05.glob("*.json")) + list((m05 / "exercises").glob("*.json")):
        info = json.loads(meta.read_text())
        row = cost[cost["item"].str.contains(f"{info['kind']} latent surrogate, seed {info['seed']} ")]
        assert len(row) == 1 and row.iloc[0]["seconds_mean"] == pytest.approx(info["seconds"])
        assert f"({info['epochs']} epochs" in row.iloc[0]["item"]


def test_requirements_are_pinned(repo):
    for fname in ("requirements.txt", "requirements-ci.txt"):
        lines = [l.strip() for l in (repo / fname).read_text().splitlines() if l.strip() and not l.startswith("#")]
        assert lines and all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[^=]+", l) for l in lines), fname


def test_no_course_wording_in_text(repo, notebooks):
    texts = {"README.md": (repo / "README.md").read_text(), "docs/REPO_MAP.md": (repo / "docs/REPO_MAP.md").read_text()}
    for stage, nb in notebooks.items():
        texts[stage] = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for name, text in texts.items():
        assert not COURSE_WORDS.search(text), f"{name}: {COURSE_WORDS.search(text).group(0)!r}"
    names = [str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts and "venv" not in p.parts]
    assert not [n for n in names if COURSE_WORDS.search(n)]


def test_no_standalone_scripts_to_smoke_test(repo):
    """The repository has no command-line scripts: the only Python outside src/ and tests/ would be one."""
    stray = [p for p in repo.rglob("*.py") if not any(part in ("src", "tests", ".git", "venv", ".venv") for part in p.parts)]
    assert stray == [], stray
