# System Identification for Quantitative Systems Pharmacology models
Title: Data-Driven System Identification in Cancer Systems Biology: A Multiscale Modeling Approach to Melanoma
Authors: Chase Christenson, Siva Viknesh, Robert L. Judson-Torres, Amirhossein Arzani

Self-contained scripts for the manuscript. These implement ADAM-SINDy with AIC-based model selection to identify sparse ODE models for a multiscale melanoma signaling model.

## Files

### `MelanomaModel.py`
Defines the two ODE systems used to generate synthetic training data:

- **`MAPKModel`** — 18-state sub-cellular MAPK signaling cascade (RAS → RAF → MEK → ERK, plus PI3K/AKT, PKA/cAMP/MITF branches). Accepts time-varying drug inputs (RAF, MEK, ERK, RAS inhibitors) and external inputs (RTK, GPCR, PPTASE). Implemented as a `scipy.integrate.solve_ivp`-compatible callable.
- **`CellModel`** — 3-state cell phenotype model (Sensitive, Differentiated, Resistant). Driven by MAPK protein outputs (ERK, PI3K, AKT, MITF). Growth and transition terms use Hill functions with deviation from steady-state as the driving signal.
- **`RandomizeMAPKParams` / `RandomizeCellParams`** — sample model parameters from biologically-motivated uniform distributions for robustness testing.
- Plotting utilities: `plotFullNetwork`, `plotCells`.

### `NetworkModel_ADAMSINDy_to_AIC.py`
Identifies the MAPK network ODEs from simulated data. Workflow:

1. **Generate data** — simulate `MAPKModel` under 4 perturbation conditions (RAF inhibitor pulse, RTK/GPCR/PPTASE stimulation steps) to produce multi-experiment time-series.
2. **Build candidate library** — terms include: constant, linear state terms, Michaelis-Menten-style terms (linear × 1/(K+x)), bilinear products, drug interaction terms.
3. **ADAM-SINDy** — jointly optimizes sparse coefficients (L1+L2 regularization via learned sparsity weights) and nonlinear Hill parameters using two Adam optimizers with step-decay schedules.
4. **AIC model selection** — enumerates all subsets (up to size 4) of nonzero terms identified by ADAM-SINDy, refits each with `scipy.optimize.least_squares`, and selects the model minimizing AIC.

Default settings: `T=60`, `dt=0.5`, 25,000 epochs, identifies one state at a time (`model_idx = [0]` for iRAS).

### `CellModel_ADAMSINDy_to_AIC.py`
Identifies the cell phenotype ODEs from simulated data. Same ADAM-SINDy → AIC workflow as the network script, but with a cell-specific candidate library:

- **S and R states**: proliferation terms `S*(1 - T/θ)*Hill(protein)` and transition terms `S*Hill(|Δprotein|)*sigmoid(±Δprotein)`.
- **D state**: transition-in terms only (`S*Hill(|Δprotein|)*sigmoid(Δprotein)`).

Perturbation: a single RAF-inhibitor sigmoid pulse; protein trajectories from `MAPKModel` are fed as inputs. Default: `T=100`, `dt=0.05`.

## Running

```bash
cd GITHUB_SINDY
python NetworkModel_ADAMSINDy_to_AIC.py   # MAPK network identification
python CellModel_ADAMSINDy_to_AIC.py      # Cell phenotype identification
```

Both scripts are self-contained and default to CPU (`torch.device('cpu')`). GPU is also supported but will be slower due to the complex candidate library structure. With `plot = True` (default), figures are displayed interactively at each stage.

To identify a different state variable, change `model_idx` at the top of the script (e.g., `model_idx = [0]` for the first state, `[0,1,2]` for all three cell states simultaneously).

## Dependencies

```
numpy torch scipy matplotlib scikit-learn
```
