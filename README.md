# Reduced-order models of airfoil flow fields and a latent-space design search

This repository holds the last two stages of a project that starts in `airfoil-cfd-surrogate-drl`, where I built a coefficient surrogate from AirfRANS and trained a flap controller inside it. Here I go from the two coefficients to the whole flow field. Stage 5 builds two reduced-order models of the AirfRANS fields of four-digit NACA airfoils interpolated onto a regular grid: a linear one from proper orthogonal decomposition with eight modes and a nonlinear one from a β-variational autoencoder with eight latent coordinates. A surrogate then maps the airfoil parameters and operating conditions to the latent coordinates, an aerodynamic head maps the latent coordinates to lift and drag, and the chain of the two screens a grid of candidate airfoils for the best lift-to-drag ratio. The notebook answers eight questions about the models and adds four sections on ensemble spread over the design space, what the latent coordinates encode, the cost of a query, and the place of the chain in a design workflow. Stage 6 puts that chain in place of the coefficient surrogate inside the flap-control environment of Stage 4 and measures what changes for the controller. Wherever a result depends on a random initialisation I repeat it with three seeds.

Portfolio page: https://vamsigudipati.github.io/airfoil-rom-betavae-design.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/05_airfrans_rom_betavae_design.ipynb` | Stage 5 — Reduced-order models of the flow fields and a latent-space design search | `results/05_airfrans_rom_betavae_design/` | `models/05_airfrans_rom_betavae_design/` | `animations/05_*` |
| `notebooks/06_field_model_in_control_loop.ipynb` | Stage 6 — The field model inside the flap-control loop | `results/06_field_model_in_control_loop/` | `models/06_field_model_in_control_loop/` | — |

Each notebook is also exported as HTML next to it, with all outputs. `results/` holds the tables (CSV) and figures (PNG); `models/05_…/` holds the POD basis, the β-VAE encoder and decoder, the parameter scaler, the latent surrogates and aerodynamic heads for both reduced-order models (seed 2026 at the top level, the other seeds and the extra β-VAEs in `exercises/`) and the scalers of the chain; `models/06_…/` holds the PPO agents trained in Stage 6 with their training logs; `animations/` holds the β-VAE training video and its final frame.

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

## Data

The fields come from the AirfRANS dataset (Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions*, NeurIPS 2022, https://airfrans.readthedocs.io/): the steady RANS solutions of the four-digit NACA family, interpolated onto a regular grid around the airfoil with five channels per case (velocity components, pressure, turbulent viscosity and the signed distance to the surface) and a train/test split. The coefficient table `data/airfrans_4digit_coefficients.csv` (one row per simulation with the Reynolds number, the angle of attack, the NACA parameters and the lift and drag coefficients) and the coefficient surrogate `data/airfrans_cl_cd_surrogate.keras` with its scalers `data/airfrans_cl_cd_scalers.joblib` are the ones built and used in the first repository.
