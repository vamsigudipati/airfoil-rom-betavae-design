# Repository map

What every file does, the order the pipeline runs in, where the numbers quoted in the README come
from, and which notebook cell each function of `src/airfoil_rom/` was lifted from. Cell numbers are
the zero-based positions in the notebook JSON (the same numbers the notebooks' own text uses).

## 1. Files

| Path | What it is |
| --- | --- |
| `README.md` | Project summary, setup, data provenance, reproduction commands, headline results, limitations. |
| `requirements.txt` | The frozen laptop environment (macOS, Python 3.12) the notebooks ran in. Pinned. |
| `requirements-ci.txt` | The minimal pinned subset the tests and CI need (same versions, CPU wheels). |
| `pyproject.toml` | Package metadata for `src/airfoil_rom` and the pytest configuration (`pythonpath = src`). |
| `notebooks/05_airfrans_rom_betavae_design.ipynb` | **Stage 5.** POD and β-VAE reduced-order models of the AirfRANS fields, latent surrogate, aerodynamic head, design search, eight questions, four additions. 88 cells, all outputs saved. |
| `notebooks/06_field_model_in_control_loop.ipynb` | **Stage 6.** The Stage 5 chain inside the flap-control environment of Stage 4: laws compared, NumPy forward pass, PPO cross-tests, decoded pressure fields, extrapolation probe. 24 cells, all outputs saved. |
| `notebooks/*.html` | nbconvert exports of the two notebooks with all outputs; what the portfolio page links to. |
| `data/airfrans_4digit_coefficients.csv` | 489 rows (378 train, 111 test): name, split, Re, alpha, NACA digits m/p/t, C_L, C_D. Built in the companion repository `airfoil-cfd-surrogate-drl`. |
| `data/airfrans_cl_cd_surrogate.keras`, `data/airfrans_cl_cd_scalers.joblib` | The Stage 3 coefficient surrogate (parameters → C_L, C_D) and its input/output scalers, from the companion repository. Used as the reference law in both notebooks. |
| `release_assets/` (git-ignored) | `airfrans_naca4_train.npz` (45.2 MB) and `airfrans_naca4_test.npz` (13.3 MB) from GitHub release `v1.0`: fields `(N, 5, 64, 128)`, parameters `(N, 5)`, names, `x_grid`, `y_grid`, `channel_names`. |
| `models/05_airfrans_rom_betavae_design/` | Everything Stage 5 fits with seed 2026: `channel_statistics.npz`, `pod_pca_8.joblib`, `beta_vae_{encoder,decoder}.keras`, `parameter_scaler.joblib`, `latent_surrogate_{pod,vae}_seed2026.{keras,json}` (the JSON holds the target scale, epochs, best epoch and training seconds), `aerodynamic_head_{pod,vae}_seed2026.keras` with `_latent_scaler.joblib`, `chain_scalers.joblib` (parameter scaler, coefficient scaler, latent scales, unit note). |
| `models/05_airfrans_rom_betavae_design/exercises/` | The same surrogates and heads for seeds 2027 and 2028, and the two extra β-VAEs (seeds 2027, 2028) of question 2. The folder name is the one the notebook writes to; its contents are the seed-repeat models. |
| `models/06_field_model_in_control_loop/` | Five PPO agents (`ppo_surrogate_seed{1,2}.zip` trained in the Stage 3 surrogate environment, `ppo_l12chain_seed{0,1,2}.zip` trained in the Stage 5 chain environment) and their `_monitor.csv` training logs (418 episodes, about 11 s of wall time each). The sixth agent (Stage 3 environment, seed 0) is Stage 4's first experiment, loaded from the companion repository. |
| `results/05_airfrans_rom_betavae_design/` | 24 files, `ex1_…` to `ex11_…`: one CSV (and PNG where there is a figure) per question or addition. Written by cells 51–82. |
| `results/06_field_model_in_control_loop/` | 12 files, `A_…` to `E_…`: one CSV/PNG per part of the Stage 6 notebook. Written by cells 14–22. |
| `animations/05_beta_vae_training_dashboard.mp4`, `…_final_frame.png` | Training video of the β-VAE (52 snapshots, 64 frames) and its last frame. The `.pkl` of snapshots is git-ignored. |
| `src/airfoil_rom/` | The pure functions of the notebooks as an importable package (section 4). |
| `tests/` | 44 pytest tests: unit tests of the package, parity tests against the notebook cells, repository smoke tests (section 5). |
| `.github/workflows/tests.yml` | GitHub Actions: Python 3.12 on `ubuntu-latest`, CPU, `pip install -r requirements-ci.txt`, `pytest`. |
| `docs/REPO_MAP.md` | This file. |

There are no command-line scripts; the whole pipeline lives in the two notebooks.

## 2. Pipeline order

```
companion repository (airfoil-cfd-surrogate-drl, Stages 1-4)
   ├── data/airfrans_4digit_coefficients.csv          (coefficient table, 489 cases)
   ├── data/airfrans_cl_cd_surrogate.keras + scalers  (Stage 3 coefficient surrogate)
   └── …/04_…/exercises/ex1_ppo.zip                   (Stage 4 PPO agent, read by Stage 6 only)
release v1.0: release_assets/airfrans_naca4_{train,test}.npz  (interpolated fields)
   │
   ▼
notebooks/05_airfrans_rom_betavae_design.ipynb   (run from notebooks/)
   cells 3-8    imports, seed 2026, load fields and parameters, sanity checks
   cells 10-14  inspect one case, plot channels
   cell 16      per-channel standardisation (training statistics)
   cell 18-21   POD (PCA, 8 modes, randomised solver)            → pod_test_mse 0.0557
   cells 23-27  β-VAE (8 latents, β warm-up, 100 epochs; retrained on every run, ~73 s)   → vae_test_mse 0.0374
   cell 29      parameters → log10 Re → StandardScaler
   cells 30, 34 model builders for the latent surrogate and the aerodynamic head (definitions only)
   cell 32      load C_L, C_D for the 489 cases
   cell 38      save the fitted objects to models/05_…, snapshots to animations/
   cells 40-44  animation + answers A1-A3
   cell 46      train or reload latent surrogates (seed 2026)   → reduction / surrogate / total MSE
   cell 48      train or reload aerodynamic heads, chain_predict, Stage 3 surrogate as reference
   cells 51-73  questions 1-8 → results/05_…/ex1_… to ex8_…
   cells 76-83  additions D1-D3 → ex9_…, ex10_…, ex11_…
   cells 84-87  workflow table, summary table, list of files written
   │
   ▼
notebooks/06_field_model_in_control_loop.ipynb   (run from notebooks/; needs the companion repository next to this one)
   cells 2-12   Stage 4 code copied verbatim: Stage 3 surrogate, NumPy forward pass, gust, environment, episode helpers, trusted α range
   cell 14      load the Stage 5 chain (seed 2026), part A: three laws at the operating point → A_*.csv/png
   cell 16      NumPy forward pass of the chain (l12_fast), check vs Keras, timing → B_timing.csv; Stage5ChainEnv
   cell 18      train or reload six PPO agents, cross-test on five gusts → C_*.csv
   cell 20      decode pressure fields along one flight → D_*.csv/png
   cell 22      shared training ranges, probe at 18° → E_*.csv/png
```

If the saved models are present, Stage 5 retrains only POD and the seed-2026 β-VAE (the extra
β-VAEs, the surrogates and the heads are reloaded), and Stage 6 trains nothing.

## 3. Where the README's numbers come from

Every number quoted in `README.md` is traced here to a results file or a notebook output cell.

| README statement | Source | Value there |
| --- | --- | --- |
| 378 training and 111 test simulations, 5 channels on a 64 × 128 grid | Stage 5 cell 7 output (`Training fields: (378, 5, 64, 128)`, `Test fields: (111, 5, 64, 128)`); `data/airfrans_4digit_coefficients.csv` (489 rows) | 378 / 111 / 5 / 64 × 128 |
| eight POD modes, eight latent coordinates | Stage 5 cell 18 (`LATENT_DIM = 8`) and `models/05_…/pod_pca_8.joblib` | 8 |
| three seeds | Stage 5 cell 46 (`SEEDS = [2026, 2027, 2028]`); Stage 6 cell 3 (`PPO_SEEDS = [0, 1, 2]`) | 3 |
| field files 58 MB together | GitHub release `v1.0` assets: 45,243,919 + 13,291,367 bytes | 58.5 MB |
| Python 3.12 | Stage 6 kernel spec (`py312-tf-sb3`); `requirements.txt` | 3.12 |
| Stage 5 retrains POD and the β-VAE, about two minutes | β-VAE training 73 s (Stage 5 cell 83, answer D3); POD, animation rendering and the surrogate reloads add the rest | ~2 min |
| Stage 6 reloads five PPO agents | `models/06_…/*.zip` (5 files); Stage 6 cell 18 output | 5 (+1 from the companion repository) |
| POD 8 modes: standardised test MSE 0.0557, 86.4 % of the training variance | `results/05_…/ex1_pod_modes.csv` row `modes=8` (`test_mse` 0.05573, `variance_captured_train` 0.86435); Stage 5 cell 18 output | 0.0557 / 86.4 % |
| 33 modes for 95 %, 128 for 99 % of the variance | `ex1_pod_modes.csv` rows 33 and 128; Stage 5 cell 51 output | 33 / 128 |
| β-VAE 8 latents: 0.0375 ± 0.0022 over three seeds | `ex2_pod_vs_vae_summary.csv` rows `beta-VAE mean (3 seeds)` 0.03747 and `beta-VAE std (3 seeds)` 0.00217 | 0.0375 ± 0.0022 |
| the surrogate adds 0.0013–0.0014 to the field error | `ex4_error_split_summary.csv` rows `MLP, mean of 3 seeds`: `surrogate_mse` 0.00135 (POD), 0.00132 (β-VAE) | 0.0014 / 0.0013 |
| β-VAE chain: C_L RMSE 0.0212 ± 0.0023, C_D RMSE 0.00074 | `ex5_aerodynamic_head_summary.csv` row `chain via beta-VAE, MLP head`: `CL_RMSE` 0.021219, `CL_RMSE_std` 0.00229, `CD_RMSE` 0.000741 | 0.0212 ± 0.0023 / 0.00074 |
| Stage 3 coefficient surrogate on the same cases: 0.0259 and 0.00087 | same file, row `coefficient surrogate of Stage 3` | 0.02593 / 0.000866 |
| POD chain: 0.0275 and 0.00104 | same file, row `chain via POD, MLP head` | 0.02749 / 0.00104 |
| every pick lies at t = 5.09 %, picks differ by model and seed | `ex6_design_search.csv` (`t_digit` 5.094 in every row; m/p digits 6.0/5.0, 6.5/4.5, 7.0/3.5 and the seed rows) | — |
| ensemble spread at the picks 6.95, 6.75, 11.69 against a grid median of 2.58 | Stage 5 cell 76 output (`Median spread … CL/CD 2.58`; picks table `CL/CD_std`) | 2.58 / 6.95 / 6.75 / 11.69 |
| 3,600 candidates, 0.091 s per batch query, 42 ms for one | `ex11_query_cost.csv` rows 0–1 (`seconds_mean` 0.09105, 0.04220) | 0.091 s / 42.2 ms |
| Stage 6: zero-gust target lift 0.7595 (Stage 3) and 0.7483 (β-VAE chain) | `results/06_…/A_summary.csv` column `target CL at 5 deg (zero gust)`; Stage 6 cell 16 output | 0.7595 / 0.7483 |
| NumPy chain 0.014 ms per call against 37–41 ms in Keras | `B_timing.csv` | 0.0140–0.0145 ms / 37.2–40.7 ms |
| PPO cross-test mean RMSE 0.0095–0.0106 for every train/test pairing, 0.14–0.15 uncontrolled | `C_cross_test_summary.csv` column `mean_RMSE mean` | 0.00952, 0.01063, 0.00957, 0.01041; 0.1496, 0.1401 |
| seed-0 agent reproduces Stage 4's 0.00979 | Stage 6 cell 18 output (`Check against … 0.00979): 0.00979`); `C_cross_test_runs.csv` row 1 | 0.00979 |
| largest C_L spread between the laws 0.050 inside the α range, 0.195 outside | Stage 6 cell 22 output; `E_sweep_outside.csv` column `CL spread` | 0.0502 / 0.1946 |
| PPO training about 11 s per agent (50,000 steps) | `models/06_…/*_monitor.csv`, last `t` column (10.75–10.99 s) | ~11 s |
| latent surrogates trained in 19–38 s, 851–1,724 epochs | `models/05_…/**/latent_surrogate_*.json` (`seconds`, `epochs`); `ex11_query_cost.csv` rows 5–10 | 18.7–38.1 s |

Numbers that appear in the notebooks' own answer cells are not repeated here; each answer cell sits
directly under the code cell whose output it quotes.

## 4. Notebook cell → package function

The notebooks keep their inline definitions, so that their saved outputs remain the record of what
ran; `src/airfoil_rom/` holds the same code, importable, and `tests/test_notebook_parity.py` executes
the notebook definitions (parsed from the JSON, never run as a notebook) against the package.

| Package | Function / class | Notebook cell | Change on extraction |
| --- | --- | --- | --- |
| `preprocessing` | `channel_statistics`, `standardise_fields`, `destandardise_fields` | 05 / 16 (and the `* channel_std[0] + channel_mean[0]` of 66, 79) | inline statements wrapped in functions |
| `preprocessing` | `to_channels_last`, `to_channels_first` | 05 / 24, 25 | inline transposes wrapped |
| `preprocessing` | `prepare_parameters` | 05 / 29 | verbatim |
| `preprocessing` | `make_candidate_grid` | 05 / 66 | the grid sizes and the operating point became arguments |
| `preprocessing` | `digits_to_chain_inputs` | 06 / 14 (`l12_params`, before the scaler) | the scaler step is left to the caller |
| `preprocessing` | `chain_inputs_to_stage3` | 05 / 48 (inside `provided_predict`) | the model call is left to the caller |
| `metrics` | `standardised_mse`, `per_channel_mse` | 05 / 18, 25, 54 | inline expressions wrapped |
| `metrics` | `coefficient_scores` | 05 / 63 (`scores`) | `test_coefficients` became an argument |
| `metrics` | `error_by_region`, `REGION_BANDS` | 05 / 57 | the loop body became a function |
| `metrics` | `nearest_distance` | 05 / 72 (`nearest`) | `S_train` became an argument |
| `metrics` | `timed` | 05 / 82 | verbatim |
| `metrics` | `episode_metrics` | 06 / 10 | verbatim |
| `models` | `get_sampling_layer` (`Sampling`), `build_beta_vae`, `beta_vae_loss` | 05 / 23 | verbatim; `Sampling` is built inside a function so that TensorFlow loads lazily |
| `models` | `beta_schedule` | 05 / 24 | the warm-up expression of the training loop |
| `models` | `build_latent_surrogate`, `build_aerodynamic_head` | 05 / 30, 34 | verbatim |
| `models` | `dense_stack`, `run_stack` | 06 / 16 (also 06 / 6) | verbatim |
| `models` | `NumpyChain` | 06 / 16 (`l12_fast`) | the eight module globals became constructor arguments |
| `control` | `one_minus_cosine_gust`, `AirfoilSurrogateGymEnv`, `ZeroPolicy`, `run_episode`, `fly`, `suite_metrics`, `SUITE` | 06 / 7, 8, 9, 12 | the aerodynamic law is an `aero_fn` argument instead of the global `predict_airfrans_aero`; `suite_metrics` takes `trust_range`. `Stage5ChainEnv` of cell 16 is `AirfoilSurrogateGymEnv(aero_fn=NumpyChain(...))` |
| `plotting` | `plot_channels`, `plot_field_comparison` | 05 / 12, 21, 27 | return the figure instead of `plt.show()` |

Left in the notebooks on purpose: the training loops (cells 24, 46, 48, 54 and 06/18), the
animation dashboard (cell 40), and every cell that reads or writes files. They are not pure and they
are what the saved outputs document.

## 5. Tests

| File | What it checks |
| --- | --- |
| `tests/test_preprocessing.py` | Shapes and round trips of the standardisation, the channel layouts, the log10 of Re, the candidate grid covering its box, the digit conversions. |
| `tests/test_metrics.py` | MSE on known shifts, RMSE/MAE/R² on a hand-computed case, region shares summing to one, nearest distances, the timing helper, episode metrics on a known case. |
| `tests/test_models.py` | β-VAE forward pass on a 2-sample batch (CPU), the sampling layer's mean and spread, the loss on known values, the β warm-up, surrogate and head parameter counts, NumPy chain against Keras. |
| `tests/test_control.py` | Gust profile, Gymnasium's `check_env`, equilibrium target lift, episode length, first-order flap response, the suite rows. |
| `tests/test_plotting.py` | The two figure helpers build the expected panels (Agg backend). |
| `tests/test_notebook_parity.py` | Definitions parsed out of notebook cells 23, 29, 30, 34, 57, 63, 72, 82 (Stage 5) and 7–12, 16 (Stage 6) give the same results as the package on synthetic inputs, including identical environment trajectories. |
| `tests/test_repo_smoke.py` | Both notebooks validate as nbformat 4, every code cell has an execution count and no error output, every results/model/data file the notebooks write or read exists, the surrogate JSON metadata agrees with `ex11_query_cost.csv`, both requirements files are fully pinned, no teaching-context wording in README, notebook text or file names, and there are no standalone scripts (hence no argument parsers to smoke-test). |

## 6. Known text inconsistencies left as they are

- The Stage 5 notebook's text cells 75, 81 and 85 say `trained_models/`; the code (cell 38) and the
  repository use `models/05_airfrans_rom_betavae_design/`. The saved outputs match the code.
- Identifiers `L12`, `l12_params`, `l12_fast`, `l12chain` (Stage 6) and the file name
  `A_polars_at_10_4_operating_point.csv` are opaque labels from an earlier numbering of the stages.
  They are referenced by the notebook code and by saved model/result file names, so they were left
  unchanged rather than re-executing the notebooks.
