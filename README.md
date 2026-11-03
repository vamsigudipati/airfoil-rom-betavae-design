# Reduced-order models of airfoil flow fields and a latent-space design search

This repository holds the last two stages of a project that starts in `airfoil-cfd-surrogate-drl`, where I built a coefficient surrogate from AirfRANS and trained a flap controller inside it. Here I go from the two coefficients to the whole flow field. Stage 5 builds two reduced-order models of the AirfRANS fields of four-digit NACA airfoils interpolated onto a regular grid: a linear one from proper orthogonal decomposition with eight modes and a nonlinear one from a β-variational autoencoder with eight latent coordinates. A surrogate then maps the airfoil parameters and operating conditions to the latent coordinates, an aerodynamic head maps the latent coordinates to lift and drag, and the chain of the two screens a grid of candidate airfoils for the best lift-to-drag ratio. The notebook answers eight questions about the models and adds four sections on ensemble spread over the design space, what the latent coordinates encode, the cost of a query, and the place of the chain in a design workflow. Stage 6 puts that chain in place of the coefficient surrogate inside the flap-control environment of Stage 4 and measures what changes for the controller. Wherever a result depends on a random initialisation I repeat it with three seeds.

Portfolio page: https://vamsigudipati.github.io/airfoil-rom-betavae-design.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/05_airfrans_rom_betavae_design.ipynb` | Stage 5 — Reduced-order models of the flow fields and a latent-space design search | `results/05_airfrans_rom_betavae_design/` | `models/05_airfrans_rom_betavae_design/` | `animations/05_*` |
| `notebooks/06_field_model_in_control_loop.ipynb` | Stage 6 — The field model inside the flap-control loop | `results/06_field_model_in_control_loop/` | `models/06_field_model_in_control_loop/` | — |

Each notebook is also exported as HTML next to it, with all outputs. `results/` holds the tables (CSV) and figures (PNG); `models/05_…/` holds the POD basis, the β-VAE encoder and decoder, the parameter scaler, the latent surrogates and aerodynamic heads for both reduced-order models (seed 2026 at the top level, the other seeds and the extra β-VAEs in `exercises/`) and the scalers of the chain; `models/06_…/` holds the PPO agents trained in Stage 6 with their training logs; `animations/` holds the β-VAE training video and its final frame.

`src/airfoil_rom/` holds the pure functions the notebooks define inline (preprocessing, metrics, model builders, the flap-control environment, plotting helpers) as an importable package, `tests/` checks them against the notebook cells and the saved results, and `.github/workflows/tests.yml` runs the tests on every push. `docs/REPO_MAP.md` lists every file, the order the pipeline runs in, the cell each function came from, and the source of every number quoted below.

## Main results

All numbers are read from `results/` or from the notebooks' output cells; `docs/REPO_MAP.md` gives the file and row for each.

| What | POD (8 modes) | β-VAE (8 latents) | Reference |
| --- | --- | --- | --- |
| Standardised test MSE of the reconstruction (111 unseen fields) | 0.0557 | 0.0375 ± 0.0022 (3 seeds) | 33 modes hold 95 % of the training variance, 128 hold 99 %; 8 modes hold 86.4 % |
| Error the latent surrogate adds to the field | 0.0014 | 0.0013 | a linear fit adds 0.06, the training mean 0.33–0.39 |
| C_L RMSE / C_D RMSE of the chain on the 111 test cases | 0.0275 / 0.00104 | 0.0212 ± 0.0023 / 0.00074 | Stage 3 coefficient surrogate 0.0259 / 0.00087 |
| Best C_L/C_D at Re 4×10⁶, α 5° (3,600 candidates) | m 6.5, p 4.5, t 5.09 | m 6.0, p 5.0, t 5.09 | every pick lies on the thin edge of the training box; other seeds pick other cambers |
| Ensemble spread of C_L/C_D at the picks (3 seeds) | — | 6.95 and 6.75 against a grid median of 2.58 | the spread is largest where the optima lie |
| Cost of a query on the laptop CPU | — | 0.091 s for 3,600 candidates, 42 ms for one | β-VAE training 73 s, latent surrogates 19–38 s |

In Stage 6 the β-VAE chain replaces the coefficient surrogate inside the flap-control environment: the zero-gust target lift moves from 0.7595 to 0.7483, the NumPy forward pass of the chain costs 0.014 ms per call against 37–41 ms in Keras, and PPO agents trained in either environment track lift in the other with the same mean RMSE (0.0095–0.0106 over the five gusts, 0.14–0.15 uncontrolled). The seed-0 agent reproduces Stage 4's score of 0.00979 exactly. Outside the shared training range of the angle of attack the two laws drift apart (largest C_L spread 0.050 inside, 0.195 outside).

## Setting up and running

I ran everything on a laptop CPU (Apple M3 Max) with Python 3.12. The frozen environment is in `requirements.txt`; the video needs `ffmpeg` on the path.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name airfoil-rom
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 05_airfrans_rom_betavae_design.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. Before running, place the files below.

**Field data (not in the git history).** The two interpolated field files `airfrans_naca4_train.npz` and `airfrans_naca4_test.npz` (58 MB together) are attached to the repository's first release. Download them into `release_assets/` (the folder is ignored by git); Stage 5 reads both, Stage 6 reads them for the grid and the training ranges.

**Files that Stage 6 takes from the first repository.** Stage 6 loads the PPO agent of Stage 4's first experiment from the companion repository, so clone `airfoil-cfd-surrogate-drl` next to this one; the notebook reads `../../airfoil-cfd-surrogate-drl/models/04_airfoil_gust_control_cfd_surrogate/exercises/ex1_ppo.zip` (if the two repositories sit elsewhere, change `R104` in the setup cell of the notebook). The Stage 5 models it needs are in `models/05_airfrans_rom_betavae_design/` of this repository, and the coefficient surrogate it compares against is in `data/`.

Stage 5 retrains the POD basis and the β-VAE when it runs (about two minutes on the laptop; the training is deterministic on the CPU) and reloads the latent surrogates, the heads and the extra β-VAEs from `models/` when they are present. Stage 6 reloads its five PPO agents from `models/06_…/` and trains nothing when they are present.

## Reproduce

Everything below is CPU only; the wall times are those recorded on the laptop (Apple M3 Max) in the notebooks' own timing cells, and they will be longer on a slower machine.

```bash
# 1. environment (once)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name airfoil-rom

# 2. field data (once): the two files of release v1.0 into release_assets/
mkdir -p release_assets
curl -L -o release_assets/airfrans_naca4_train.npz https://github.com/vamsigudipati/airfoil-rom-betavae-design/releases/download/v1.0/airfrans_naca4_train.npz
curl -L -o release_assets/airfrans_naca4_test.npz  https://github.com/vamsigudipati/airfoil-rom-betavae-design/releases/download/v1.0/airfrans_naca4_test.npz

# 3. Stage 5: POD and the seed-2026 beta-VAE are retrained (73 s for the beta-VAE), everything else is reloaded from models/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 05_airfrans_rom_betavae_design.ipynb

# 4. Stage 6: needs ../../airfoil-cfd-surrogate-drl (the companion repository) for Stage 4's agent; the five other agents are reloaded
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 06_field_model_in_control_loop.ipynb
cd ..

# 5. the tests (no field data needed): about 10 s once the packages are installed
pip install -r requirements-ci.txt
python -m pytest -q
```

Expected wall time on the laptop CPU: Stage 5 about 3 minutes with the saved models present (β-VAE training 73 s, POD a few seconds, the animation render and the analysis cells the rest); add about 20–40 s per latent surrogate and 73 s per extra β-VAE if `models/05_…` is emptied. Stage 6 about 2 minutes with the saved agents (the cross-tests run the NumPy forward pass); add about 11 s per PPO agent (50,000 steps) if `models/06_…` is emptied. The β-VAE training is deterministic on the CPU with the seed set in the notebook; a re-run on another machine or TensorFlow version can differ in the last digits.

## Data

The fields come from the AirfRANS dataset (Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions*, NeurIPS 2022, https://airfrans.readthedocs.io/): the steady RANS solutions of the four-digit NACA family, interpolated onto a regular grid around the airfoil with five channels per case (velocity components, pressure, turbulent viscosity and the signed distance to the surface) and a train/test split. The coefficient table `data/airfrans_4digit_coefficients.csv` (one row per simulation with the Reynolds number, the angle of attack, the NACA parameters and the lift and drag coefficients) and the coefficient surrogate `data/airfrans_cl_cd_surrogate.keras` with its scalers `data/airfrans_cl_cd_scalers.joblib` are the ones built and used in the first repository.

## Limitations

- **One dataset, one family.** 489 steady RANS cases of four-digit NACA sections on a 64 × 128 grid, Re 2–6 × 10⁶, α −4.9° to 14.8°. No stall, no unsteady flow, no flap in the fields: Stage 6's decoded pressure fields are steady fields at an equivalent angle of attack.
- **Small seed sample.** Three seeds separate POD from the β-VAE and the chains from one another, but they do not pin any error more finely than their spread, and the ensemble of Stage 5's addition D1 shares one β-VAE, so its spread is a lower bound on the real disagreement.
- **The design search extrapolates along an edge.** Every best design lies at the thinnest section in the data (t = 5.09 %), the pick changes with the seed, and no simulation confirms any of them.
- **Errors concentrate at the surface.** Within 0.02 chord of the body the reconstruction error is two to three orders of magnitude larger than beyond 0.1 chord, for both models.
- **Stage 6 is a sensitivity study, not a validation.** The two aerodynamic laws come from the same 378 training cases and disagree little inside that range; the controller comparison says the chain is a drop-in replacement there, not that either law is right outside it.
- **Reproducibility.** The saved models and results are from one machine (Apple M3 Max, TensorFlow 2.21, CPU). The tests check the code paths and the consistency of the saved files, not the training runs themselves.
