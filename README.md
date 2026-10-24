# Sliced Inverse Regression Revisited — Simulation and Real Data Code

Companion code for:

> Asrir, N. *Sliced Inverse Regression Revisited: A Comprehensive Review with
> Simulation and Real Data Applications.* (Submitted to *International
> Statistical Review*.)

This repository contains the R scripts used to produce every simulation
result, table, and figure in the paper.

## Files

| File | Purpose |
|---|---|
| `simulation_framework.R` | Defines the five estimator wrappers (SIR, SAVE, LassoSIR, SIR-NCT, SIR-WLS), the data-generating process (link functions, correlation designs), the main simulation loop (`run_simulation()`), the `p`-scaling grid (`run_p_scaling()`), the dimension-estimation sub-study (`run_dimension_substudy()`), and the plotting functions used to generate the paper's figures. |
| `real_data_applications.R` | Applies the same five estimators to the *Big Mac* (low-dimensional) and *Riboflavin* (high-dimensional, `p >> n`) datasets, with paired, repeated cross-validation (`run_real_data()`). Sources `simulation_framework.R` for the estimator wrappers. |

## Dependencies

Install the required packages once:

```r
install.packages(c("dr", "Rdimtools", "glmnet", "mvtnorm", "LassoSIR",
                    "ggplot2", "devtools"))
devtools::install_github("nasrir-ai/SIR-WLS")
devtools::install_github("nasrir-ai/nctsir", subdir = "nctsir 2")
```

`sirwls` (companion package to Asrir & Mkhadri, 2025, *Journal of Statistical
Computation and Simulation*) provides `wls.sir()`, `X_rand()`, `TPR.funct()`,
`FPR.funct()`, and `costeta()` — the SIR-WLS estimator and the simulation
utilities (predictor generation, proximity/variable-selection metrics) used
throughout both scripts. `nctsir` (companion package to Asrir & Mkhadri, 2025,
*Communications in Statistics – Theory and Methods*) provides `sir.nct()`,
the SIR-NCT estimator.

## Data

Both real-data examples use public, third-party R packages — no data files
are included in this repository:

- *Big Mac* (`alr4::BigMac2003`, Union Bank of Switzerland, *Prices and
  Earnings Around the Globe*, 2003) — `n = 69`, `p = 9`.
- *Riboflavin* (`hdi::riboflavin`) — `n = 71`, `p = 4088`.

## Running

```r
source("simulation_framework.R")   # main simulation study + dimension-
                                    # estimation sub-study + plotting
                                    # functions (loads all dependencies)
source("real_data_applications.R") # Big Mac and Riboflavin applications
```

Each script contains example calls (commented out near the bottom) showing
the exact settings (`n`, `p`, number of replications, etc.) used to produce
the results reported in the paper. All random seeds are set within the
scripts, so results are reproducible.

Some cells (in particular the `p = 500` high-dimensional simulation and the
`Riboflavin` real-data fits) involve near-singular covariance matrices and
can take significantly longer per replicate than the moderate-dimensional
cells; both scripts wrap individual fits in a wall-clock timeout so a single
slow replicate cannot hang the whole run.

## Citation

If you use this code, please cite the paper above, along with the `sirwls`
and `nctsir` packages if you use the SIR-WLS or SIR-NCT estimators
specifically.

## License

MIT
