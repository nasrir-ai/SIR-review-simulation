## =============================================================================
## simulation_framework.R
##
## Comparative simulation study for:
##   "Sliced Inverse Regression Revisited: A Comprehensive Review with
##    Simulation and Real Data Applications" (Asrir)
##
## Purpose
## -------
## Extends the comparison already published with SIR-WLS into a systematic
## 3-factor design that targets the two most consequential open problems
## flagged in the review's literature synthesis:
##   (1) sensitivity to the LINK FUNCTION (linear / cubic / exponential /
##       non-monotone sine), and
##   (2) reliability of STRUCTURAL DIMENSION (d) estimation, compared across
##       a custom ladle-estimator implementation and the sequential
##       chi-squared test of Li (1991) / Schott (1994).
##
## REVISION NOTE (this version)
## -----------------------------
## This version was rewritten against the actual function definitions found
## in `myfunctions.R` (rather than assumed signatures). Concretely:
##   - `LassoSIR()` is NOT taken from myfunctions.R (its `LassoSIR.v2` is
##     unused here). It is called directly from the installed `LassoSIR`
##     package: `LassoSIR(X, y, H, no.dim = 1, solution.path = FALSE,
##     categorical = FALSE, nfolds = 10, screening = FALSE)`, which returns
##     a list with a `$beta` matrix -- this avoids the free-variable bug in
##     `LassoSIR.v2` described further down.
##   - `FDR.funct()` does not exist in myfunctions.R (only `TPR.funct`,
##     `FPR.funct`, `FNR.funct`, `TNR.funct`, plus an index-based
##     `compute_TPR_FPR()`). A local `fdr_funct()` is defined below,
##     mirroring the FP/(FP+TP) definition already used internally by
##     `compute_TPR_FPR()`, so the FDR reported here is consistent with the
##     rest of your codebase.
##   - `X_rand()` only implements `type` %in% c("block","autoR","homog",
##     "homog2") -- there is no "spiked" design in myfunctions.R. The design
##     grid below uses "homog2" (the sparse-signal correlation structure
##     from Lin, Zhao & Liu 2019) in place of the previously assumed
##     "spiked-eigenvalue" design.
##   - `X_rand()`'s `S` argument (needed for type = "homog2") is now always
##     passed explicitly as `S = seq_len(q)` to avoid relying on its
##     self-referential default `S = S`.
##   - `wls.sir()` and `sir.nct()` matched the previously assumed signatures
##     and are used unchanged (`wls.sir()` returns a list with `$betahat`;
##     `sir.nct()` returns the beta vector directly).
##   - (For the record: `myfunctions.R`'s own `LassoSIR.v2()` has a latent
##     bug -- its `solution.path = FALSE` branch references a free variable
##     `categorical` that is never defined as a formal argument or local
##     variable, which throws "object 'categorical' not found" once
##     `no.dim` is fixed. This is moot here since we now call the `LassoSIR`
##     package's own `LassoSIR()` instead, but worth fixing in
##     myfunctions.R if you use `LassoSIR.v2` elsewhere.)
##
## This script is a TEMPLATE: it is designed to run in your own R
## environment where the `sirwls` and `nctsir` packages (see below), `dr`,
## `Rdimtools`, `glmnet`, and `mvtnorm` are available (no R interpreter is
## available in the assistant session that produced this file, so it has
## been checked for syntax and for consistency against the estimators'
## actual function signatures, but not executed against real data).
##
## =============================================================================

## ---- 0. Packages & your own functions ----------------------------------------

## Install once if needed:
# install.packages(c("dr", "Rdimtools", "glmnet", "mvtnorm", "LassoSIR",
#                     "ggplot2", "devtools"))
# devtools::install_github("nasrir-ai/SIR-WLS")
# devtools::install_github("nasrir-ai/nctsir", subdir = "nctsir 2")  # confirm
#   the subdir path against the repo's actual layout before running; adjust
#   or drop `subdir` if the package's DESCRIPTION file sits at the repo root.

library(dr)
library(Rdimtools)
library(glmnet)
library(mvtnorm)
library(LassoSIR)   # provides LassoSIR() directly -- see run_lassosir() below
library(ggplot2)    # only needed for Section 9 (plots)

## Estimator/utility functions this script depends on -- wls.sir(), X_rand(),
## TPR.funct(), FPR.funct(), costeta() -- are provided by the sirwls package
## (companion to Asrir & Mkhadri, 2025, JSCS), and sir.nct() by the nctsir
## package (companion to Asrir & Mkhadri, 2024, CSTM). Both replace what was
## previously sourced from a personal, unpublished myfunctions.R.
library(sirwls)
library(nctsir)

## Safety checks -- stop early with a clear message if a required function is
## missing, rather than failing deep inside the simulation loop.
required_functions <- c("wls.sir", "sir.nct", "costeta", "TPR.funct",
                         "FPR.funct", "X_rand", "LassoSIR")
missing_fn <- required_functions[!sapply(required_functions, exists)]
if (length(missing_fn) > 0) {
  message("NOTE: the following functions are not yet loaded in this session: ",
          paste(missing_fn, collapse = ", "),
          ". Check that the sirwls, nctsir, and LassoSIR packages are ",
          "installed (see the devtools::install_github() calls above).")
}

## ---- 1. Link functions (Section: "Link function comparison") ---------------

## Four link functions spanning linear -> monotone-nonlinear -> non-monotone.
link_functions <- list(
  linear      = function(t, eps) t + eps,
  cubic       = function(t, eps) (t^3) / 2 + eps,
  exponential = function(t, eps) exp(t) + eps,
  sine        = function(t, eps) sin(pi * t / 2) + eps
)

## ---- 2. Design factor (uses the real X_rand() from myfunctions.R) -----------
##
## X_rand()'s `type` argument only supports "block", "autoR", "homog", and
## "homog2" (there is no "spiked" option in myfunctions.R). "homog2" is the
## sparse-signal correlation structure of Lin, Zhao & Liu (2019): predictors
## in the active set S and predictors outside S are each internally
## equicorrelated at `rho`, with a much weaker (0.1) cross-correlation
## between the two blocks. `S` is supplied at call time in run_simulation()
## as `S = seq_len(q)`, so it is *not* listed here.
design_grid <- list(
  block  = list(type = "block",  alpha = c(0.2, 0.3, 0.7)),
  autoR  = list(type = "autoR",  rho   = 0.5),
  homog2 = list(type = "homog2", rho   = 0.6)
)

## ---- 3. Method wrappers ------------------------------------------------------
## Each wrapper takes (X, y, ndim, ...) and returns a plain numeric vector
## (or ndim-column matrix) beta-hat, so every method can be scored with the
## same costeta() / TPR.funct() / FPR.funct() / fdr_funct() calls.

run_sir_classic <- function(X, y, ndim = 1) {
  out <- Rdimtools::do.sir(X, y, ndim = ndim)
  as.matrix(Re(out$projection))
}

run_save_classic <- function(X, y, ndim = 1) {
  out <- Rdimtools::do.save(X, y, ndim = ndim)
  as.matrix(Re(out$projection))
}

## LassoSIR(): called directly from the installed `LassoSIR` package (Lin,
## Zhao & Liu, 2019), not from myfunctions.R. Returns a list with a `$beta`
## matrix.
run_lassosir <- function(X, y, H = 10, ndim = 1) {
  fit <- LassoSIR(X, y, H, no.dim = ndim, solution.path = FALSE,
                   categorical = FALSE, nfolds = 10, screening = FALSE)
  as.matrix(fit$beta)
}

## ---- 3b. Cross-validated (K, tau) selection for SIR-NCT ---------------------
##
## DIAGNOSIS: earlier versions of run_nctsir() called sir.nct(X, y,
## k = min(n, p), tau = 0.1) -- a fixed, never-cross-validated tau, and a K
## that equals p while p < n (genuine compression) but *saturates at n* once
## p >= n. Once K = n, the "reduced" canonical space is no longer actually
## reduced relative to the sample size, so SIR-NCT's first-stage SIR fit
## inherits classical SIR's own p ~ n breakdown -- this is exactly what
## produced the collapse from proximity ~0.74 (p=50) to ~0.09 (p=100) in the
## p-scaling run: at p=50, K=min(100,50)=50 < n; at p=100 and p=300,
## K=min(100,p)=100=n, i.e. no compression at all. Not a flaw in SIR-NCT
## itself -- a flaw in this script's fixed default for K.
##
## myfunctions.R already ships nct_cv()/nct_cv_opt() for tuning, but (a)
## they only search over tau, never K, so they would not have caught this;
## (b) nct_cv() references a free variable `n` that is never defined inside
## it or passed in (relies on a global `n` existing by accident) and (c) it
## depends on the `np` package. Rather than silently patch your
## myfunctions.R, cv_select_nctsir() below is a fresh, self-contained
## implementation that cross-validates BOTH K and tau, with every candidate
## K kept strictly below n by construction, using nw() (already in
## myfunctions.R, Nadaraya-Watson kernel regression, no extra package) for
## the out-of-fold prediction step -- the same idea nct_cv() sketches (its
## commented-out nw() line) but wired up and bug-fixed.
##
## @param K_grid candidate compression dimensions; default is 4 points
##   spanning 10%-50% of min(n-1, p), so K < n always.
## @param tau_grid candidate soft-threshold levels.
cv_select_nctsir <- function(X, y, K_grid = NULL,
                              tau_grid = seq(0.05, 0.5, length.out = 5),
                              nfolds = 5, threshold = "soft") {
  n <- nrow(X); p <- ncol(X)
  if (is.null(K_grid)) {
    K_grid <- unique(pmax(2, round(seq(0.1, 0.5, length.out = 4) * min(n - 1, p))))
  }
  K_grid <- K_grid[K_grid < n & K_grid <= p]
  if (length(K_grid) == 0) stop("cv_select_nctsir(): no candidate K satisfies K < n; check n, p.")

  folds <- sample(rep(seq_len(nfolds), length.out = n))
  grid <- expand.grid(K = K_grid, tau = tau_grid)
  grid$cv_error <- NA_real_

  for (g in seq_len(nrow(grid))) {
    K <- grid$K[g]; tau <- grid$tau[g]
    fold_err <- numeric(nfolds)
    for (f in seq_len(nfolds)) {
      tr <- folds != f; te <- folds == f
      beta_hat <- tryCatch(
        as.vector(sir.nct(X[tr, , drop = FALSE], y[tr], K, tau, threshold = threshold)),
        error = function(e) rep(NA_real_, p))
      if (any(is.na(beta_hat)) || all(beta_hat == 0)) { fold_err[f] <- Inf; next }
      idx_tr <- as.vector(X[tr, , drop = FALSE] %*% beta_hat)
      idx_te <- as.vector(X[te, , drop = FALSE] %*% beta_hat)
      h <- tryCatch(stats::bw.nrd0(idx_tr), error = function(e) NA_real_)
      if (!is.finite(h) || h <= 0) h <- max(sd(idx_tr), 1e-3)
      y_pred <- tryCatch(nw(x = idx_te, X = idx_tr, Y = y[tr], h = h),
                          error = function(e) rep(mean(y[tr]), length(idx_te)))
      fold_err[f] <- mean((y[te] - y_pred)^2, na.rm = TRUE)
    }
    grid$cv_error[g] <- mean(fold_err)
  }

  best <- grid[which.min(grid$cv_error), ]
  list(K = best$K, tau = best$tau, grid = grid)
}

## sir.nct(): real signature is sir.nct(x, y, k, tho, threshold = "soft");
## returns the beta-hat vector directly (ndim is implicitly 1 -- do.sir()
## is called internally with ndim = 1).
##
## @param cv if TRUE (or if k/tau are left NULL), K and tau are selected via
##   cv_select_nctsir() above instead of using the caller-supplied fixed
##   values -- this is now the default, precisely to avoid the K=min(n,p)
##   trap described above.
run_nctsir <- function(X, y, k = NULL, tau = NULL, threshold = "soft",
                        cv = TRUE, ...) {
  if (cv || is.null(k) || is.null(tau)) {
    sel <- cv_select_nctsir(X, y, threshold = threshold, ...)
    k <- sel$K; tau <- sel$tau
  }
  as.matrix(sir.nct(X, y, k, tau, threshold = threshold))
}

## wls.sir(): real signature is
##   wls.sir(x, y, nslice = 10, cn1 = 0.1, cn2 = 1, choose.dir = FALSE,
##           categorical = FALSE, ndim = 1)
## and returns list(wls, select, betahat) with betahat a p x ndim matrix,
## zero-filled outside the screened set.
run_wlssir <- function(X, y, ndim = 1) {
  fit <- wls.sir(X, y, ndim = ndim)
  as.matrix(Re(fit$betahat))
}

## ---- 4. Scoring helpers -------------------------------------------------------
##
## costeta(), TPR.funct(), and FPR.funct() are used as defined in
## myfunctions.R. FDR.funct() does not exist there, so fdr_funct() below
## reproduces, on beta-vectors rather than index sets, the same FP/(FP+TP)
## definition already used internally by myfunctions.R's compute_TPR_FPR().
fdr_funct <- function(beta, hatbeta) {
  TP <- sum(beta != 0 & hatbeta != 0)
  FP <- sum(hatbeta != 0 & beta == 0)
  if ((FP + TP) == 0) return(0)
  FP / (FP + TP)
}

## ---- 5. SIR eigen-decomposition helper + ladle estimator (Luo & Li, 2016) ---
##
## sir_eigen(): shared helper computing the sample SIR generalized
## eigenvalues/eigenvectors of Sigma_X^{-1} Lambda. Neither myfunctions.R
## nor Rdimtools exposes SIR eigenvalues directly -- in particular,
## `Rdimtools::do.sir()` only returns `$Y`, `$trfinfo`, and `$projection`
## (no eigenvalue field at all, under any name, in any installed version).
## An earlier version of this script assumed a `$eigvalue` field that does
## not exist, which silently produced a zero-length BIC vector and crashed
## `run_dimension_substudy()` with "replacement has length zero". This
## helper is used both by `ladle_estimator()` below and by the BIC-type
## criterion in Section 7, so the two criteria are compared on eigenvalues
## computed the same way.
##
## @param X predictor matrix
## @param y response vector
## @param nslice number of slices
sir_eigen <- function(X, y, nslice = 10) {
  n <- nrow(X); p <- ncol(X)
  Xc <- scale(X, scale = FALSE)
  sl <- dr::dr.slices.arc(y, nslice)
  idx <- sl$slice.indicator; nh <- sl$slice.sizes
  Sigma <- cov(Xc) * (n - 1) / n
  slice_means <- sapply(seq_along(nh), function(h) colMeans(Xc[idx == h, , drop = FALSE]))
  Lambda <- Reduce(`+`, lapply(seq_along(nh), function(h) {
    (nh[h] / n) * tcrossprod(slice_means[, h])
  }))
  eig <- eigen(solve(Sigma + diag(1e-6, p)) %*% Lambda, symmetric = FALSE)
  list(values = Re(eig$values), vectors = Re(eig$vectors))
}

## Sequential chi-squared test for the structural dimension d (Li, 1991;
## Schott, 1994) -- used in run_dimension_substudy() below as the second
## criterion, in place of an earlier ad hoc "BIC-type" criterion.
##
## CORRECTION: an earlier version of this script scored a home-grown
## criterion there, borrowing the "BIC" name from wls.sir()'s own
## predictor-screening step -- but that BIC chooses the number of
## *retained predictors* (SIR-WLS's screening size), a different
## quantity from the structural dimension d estimated here; conflating
## the two was a modeling error, not just an implementation bug. The
## criterion itself was also defined as -log(cumsum(evals)/denom) +
## penalty(k), which necessarily diverges to -Inf as the cumulative
## ratio approaches 1 -- which happens quickly, since Lambda has rank at
## most nslice - 1 by construction -- so it was mechanically biased
## toward the largest k considered (Kmax = 8) regardless of how the
## denominator was normalized, exactly the boundary-saturation symptom
## seen in every real run so far. The actual literature-standard
## approach to this problem -- and the one already cited in
## Section~\ref{sec:order-determination} of the paper (Li 1991; Schott
## 1994) -- is a sequential chi-squared test on the smallest eigenvalues
## of Sigma_X^{-1} Lambda: under H0: d = k (only the first k eigenvalues
## are non-zero in the population),
##   Lambda_k = n * sum_{j=k+1}^{p} lambda_j  ~approx  chi^2_{(p-k)(H-1-k)}
## Testing sequentially from k = 0 upward and stopping at the first k
## that is NOT rejected gives d_hat: a genuine hypothesis test with a
## bounded statistic, rather than an information criterion borrowed from
## an unrelated screening step.
##
## @param alpha test level at each step of the sequential procedure
sir_chi2_dhat <- function(X, y, nslice = 10, alpha = 0.05) {
  n <- nrow(X); p <- ncol(X); H <- nslice
  eig <- sir_eigen(X, y, nslice = nslice)
  evals <- sort(eig$values, decreasing = TRUE)
  evals[!is.finite(evals) | evals < 0] <- 0

  Kmax_test <- min(p - 1, H - 2)   # need df = (p-k)(H-1-k) > 0
  stat_v <- pval_v <- df_v <- rep(NA_real_, Kmax_test + 1)
  d_hat <- Kmax_test
  for (k in 0:Kmax_test) {
    df <- (p - k) * (H - 1 - k)
    stat <- n * sum(evals[(k + 1):p])
    pval <- stats::pchisq(stat, df, lower.tail = FALSE)
    stat_v[k + 1] <- stat; pval_v[k + 1] <- pval; df_v[k + 1] <- df
    if (pval > alpha) { d_hat <- k; break }
  }
  list(d_hat = d_hat, stat = stat_v, pval = pval_v, df = df_v, eigenvalues = evals)
}

## Combines the (normalized) decay of eigenvalues with the bootstrap
## variability of the corresponding eigenvectors into a single criterion
## whose minimizer estimates the structural dimension d. This is a
## simplified, from-scratch reimplementation for this simulation study --
## myfunctions.R does not define a ladle estimator, and no validated CRAN
## package for it was found -- cross-check against Luo & Li (2016) before
## using results in the paper itself.
##
## @param X predictor matrix
## @param y response vector
## @param Kmax maximum candidate dimension to scan (Kmax < min(n, p))
## @param B number of bootstrap resamples for eigenvector variability
ladle_estimator <- function(X, y, Kmax = 10, B = 200, nslice = 10) {
  n <- nrow(X); p <- ncol(X)
  Kmax <- min(Kmax, p - 1, n - 2)

  base <- sir_eigen(X, y, nslice)
  ord <- order(base$values, decreasing = TRUE)
  lambda <- base$values[ord]
  V <- base$vectors[, ord, drop = FALSE]

  ## Bootstrap eigenvector variability phi_n(k), k = 0..Kmax
  phi <- numeric(Kmax + 1)
  boot_vecs <- vector("list", B)
  for (b in seq_len(B)) {
    resample <- sample(seq_len(n), n, replace = TRUE)
    eb <- tryCatch(sir_eigen(X[resample, , drop = FALSE], y[resample], nslice),
                   error = function(e) NULL)
    if (is.null(eb)) { boot_vecs[[b]] <- NULL; next }
    ob <- order(eb$values, decreasing = TRUE)
    boot_vecs[[b]] <- eb$vectors[, ob, drop = FALSE]
  }
  boot_vecs <- Filter(Negate(is.null), boot_vecs)

  for (k in 0:Kmax) {
    if (k == 0) { phi[k + 1] <- 0; next }
    Vk <- V[, seq_len(k), drop = FALSE]
    discrepancies <- sapply(boot_vecs, function(Vb) {
      Vbk <- Vb[, seq_len(k), drop = FALSE]
      # subspace distance via singular values of V_k^T Vb_k (1 = identical)
      sv <- svd(crossprod(Vk, Vbk))$d
      1 - prod(pmin(sv, 1))
    })
    phi[k + 1] <- mean(discrepancies)
  }

  ## Normalized eigenvalue term f_n(k) and combined ladle criterion g_n(k)
  lambda_trunc <- lambda[seq_len(Kmax + 1)]
  f_n <- lambda_trunc / (1 + sum(lambda_trunc))
  phi_n <- phi / (1 + sum(phi))
  g_n <- phi_n + f_n

  list(d_hat = which.min(g_n) - 1, g_n = g_n, eigenvalues = lambda_trunc,
       f_n = f_n, phi_n = phi_n, phi_raw = phi)
}

## ---- 6. Main simulation loop: design x link function x method --------------
##
## REVISION NOTE: this version adds the same three safety features
## run_p_scaling() already has (Section 6b) -- parallel::mclapply() across
## repetitions, a per-method wall-clock timeout via setTimeLimit(), and
## verbose per-cell progress messages -- since scaling `repetitions` up
## from the pilot value (5) toward the paper's intended final scale
## (repetitions = 200 at n = 500, p = 700) is exactly the kind of run that
## previously went overnight with no output and no way to tell whether it
## was progressing or stuck (see the long comment above run_p_scaling()).
## Use probe_simulation_time() below FIRST, with a small repetitions_probe
## (e.g. 2), to get a real timing estimate for your target (n, p) before
## committing to a specific `repetitions` value.
##
## @param timeout_sec per-method, per-replicate wall-clock cap (as in
##   run_p_scaling()); a failed/timed-out call is scored as a zero vector
##   rather than allowed to hang the whole run.
## @param mc.cores number of cores for parallel::mclapply() (Mac/Linux
##   only; falls back to serial on Windows or without the parallel
##   package). Defaults to all-but-one detected core.
## @param verbose if TRUE, prints one line per (design, link) cell with
##   elapsed time for that cell's `repetitions` replications.
run_simulation <- function(n = 300, p = 200, q = 10, repetitions = 50,
                            nslice = 10, timeout_sec = 120,
                            mc.cores = max(1, parallel::detectCores() - 1),
                            verbose = TRUE) {

  methods <- c("SIR", "SAVE", "LassoSIR", "SIR-NCT", "SIR-WLS")
  results <- expand.grid(design = names(design_grid),
                          link = names(link_functions),
                          method = methods,
                          stringsAsFactors = FALSE)
  results$abscos <- results$TPR <- results$FPR <- results$FDR <-
    results$Cor <- results$Time <- NA_real_

  Beta <- c(rep(1, q), rep(0, p - q))

  use_parallel <- .Platform$OS.type == "unix" && requireNamespace("parallel", quietly = TRUE) &&
    mc.cores > 1
  if (verbose) {
    message(if (use_parallel) paste0("Running in parallel on ", mc.cores, " cores.")
            else "Running serially (set mc.cores > 1 on Mac/Linux to parallelize).")
  }

  ## Wraps one method call in a wall-clock timeout; a failed/timed-out
  ## call returns a zero vector rather than propagating the error, exactly
  ## as in run_p_scaling()'s safe_run().
  safe_run <- function(expr_fun) {
    t0 <- Sys.time()
    b <- tryCatch({
      setTimeLimit(elapsed = timeout_sec, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      expr_fun()
    }, error = function(e) rep(0, p))
    list(b = b, dt = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  }

  run_and_score <- function(betahat, time_sec, X, Beta, t_index) {
    betahat <- as.vector(betahat)
    c(abscos = abs(costeta(Beta, betahat)),
      TPR = TPR.funct(Beta, betahat),
      FPR = FPR.funct(Beta, betahat),
      FDR = fdr_funct(Beta, betahat),
      Cor = if (stats::var(betahat) != 0) abs(cor(X %*% betahat, t_index)) else 0,
      Time = time_sec)
  }

  one_replicate <- function(des, link_fn) {
    ## S = seq_len(q) is passed explicitly on every call (see Section 2
    ## note); it is only used internally by X_rand() when type = "homog2"
    ## and is silently ignored for "block"/"autoR".
    X <- do.call(X_rand, c(list(n = n, p = p, q = q, S = seq_len(q)),
                            des[!(names(des) %in% "type")],
                            list(type = des$type)))
    t_index <- X %*% Beta
    eps <- rnorm(n)
    y <- as.vector(link_fn(t_index, eps))

    out <- list()
    r <- safe_run(function() run_sir_classic(X, y));                                    out$SIR      <- run_and_score(r$b, r$dt, X, Beta, t_index)
    r <- safe_run(function() run_save_classic(X, y));                                   out$SAVE     <- run_and_score(r$b, r$dt, X, Beta, t_index)
    r <- safe_run(function() run_lassosir(X, y, H = nslice));                           out$LassoSIR <- run_and_score(r$b, r$dt, X, Beta, t_index)
    r <- safe_run(function() run_nctsir(X, y, cv = TRUE));                              out$`SIR-NCT` <- run_and_score(r$b, r$dt, X, Beta, t_index)
    r <- safe_run(function() run_wlssir(X, y));                                         out$`SIR-WLS` <- run_and_score(r$b, r$dt, X, Beta, t_index)
    out
  }

  for (des_name in names(design_grid)) {
    des <- design_grid[[des_name]]

    for (link_name in names(link_functions)) {
      link_fn <- link_functions[[link_name]]
      if (verbose) message("=== design = ", des_name, ", link = ", link_name,
                            " (", repetitions, " replications) ===")
      t_cell <- Sys.time()

      acc <- if (use_parallel) {
        parallel::mclapply(seq_len(repetitions), function(i) one_replicate(des, link_fn),
                            mc.cores = mc.cores)
      } else {
        lapply(seq_len(repetitions), function(i) one_replicate(des, link_fn))
      }

      for (m in methods) {
        vals <- t(sapply(acc, function(a) a[[m]]))
        means <- colMeans(vals, na.rm = TRUE)
        rows <- which(results$design == des_name & results$link == link_name & results$method == m)
        results[rows, c("abscos", "TPR", "FPR", "FDR", "Cor", "Time")] <- as.list(means)
      }
      if (verbose) message("  done (", round(as.numeric(difftime(Sys.time(), t_cell, units = "secs")), 1), "s)")
    }
  }

  results
}

## Timing probe: runs run_simulation() with a SMALL repetitions_probe
## (default 2) at your target (n, p, q) and extrapolates the expected
## total time for larger repetitions counts, so you can decide how many
## repetitions are actually feasible BEFORE committing to a long run.
## Because the 12 design x link cells are run sequentially (each one
## internally parallelized across repetitions_probe), the extrapolation
## divides out the per-replication cost per cell rather than just scaling
## total wall time linearly, which is a closer (though still approximate)
## estimate once mc.cores > 1.
##
## @param repetitions_probe how many replications to actually run for the
##   timing estimate; 2 is usually enough to get a stable per-replicate
##   cost without spending much time on the probe itself.
## @param targets repetition counts to extrapolate to and report.
probe_simulation_time <- function(n, p, q, nslice = 10, repetitions_probe = 2,
                                   mc.cores = max(1, parallel::detectCores() - 1),
                                   targets = c(20, 50, 100, 200)) {
  message("Probing timing at n = ", n, ", p = ", p, ", q = ", q,
          " with repetitions = ", repetitions_probe, " ...")
  t0 <- Sys.time()
  invisible(run_simulation(n = n, p = p, q = q, repetitions = repetitions_probe,
                            nslice = nslice, mc.cores = mc.cores, verbose = TRUE))
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  per_rep <- dt / repetitions_probe
  message(sprintf("\nProbe total: %.1fs for %d replications (%.2fs/replication, all 12 design x link cells combined).",
                   dt, repetitions_probe, per_rep))
  for (r in targets) {
    est_sec <- per_rep * r
    message(sprintf("  repetitions = %-4d -> estimated ~ %.1f min (%.1f s)", r, est_sec / 60, est_sec))
  }
  message("\nThese are rough linear extrapolations (parallel-core overhead, memory ",
          "pressure, and swap can make the real run non-linearly slower at high ",
          "repetitions -- treat this as a lower bound, not a guarantee).")
  invisible(list(probe_dt = dt, per_rep = per_rep))
}

## ---- 6b. Dimension-scaling study: performance as a function of p -----------
##
## Unlike Sections 6 and 9's other plots (which just re-visualize the design-
## and link-comparison tables), this operationalizes something not yet
## delivered numerically: item 1 of Section 5.1 promised crossing "sample
## sizes/dimensions spanning p<n, p~n, and p>n", but run_simulation() above
## only ever varies the correlation structure at a single fixed (n, p). This
## function instead fixes n and a single design/link, and sweeps p across a
## grid that crosses n -- directly illustrating the phase-transition result
## of Lin, Zhao & Liu (2018) discussed in Section 3.2 (sqrt(n)-consistency
## requires p = o(n); breakdown as p approaches/exceeds n).
##
## Every method call is wrapped in tryCatch() *and* a hard per-call wall-clock
## timeout (see `timeout_sec` below): once p >= n, Sigma_X is singular, and
## classical (non-regularized) SIR/SAVE/SIR-NCT are expected to fail or
## degrade sharply -- that failure is the point of the plot, not a bug to
## hide, so a failed or timed-out replicate is scored as a zero vector
## (worst case) rather than dropped or left to hang the whole run.
##
## IMPORTANT -- if a previous run "ran all night with no output": the most
## likely cause is not raw compute cost (720 calls at p up to 400 should
## take minutes, not hours) but a *silent interactive prompt* somewhere
## down the call stack (e.g. a readline() left over from an installed
## package's interactive branch) blocking forever on stdin with no error
## and no timeout, which setTimeLimit() cannot always interrupt because the
## process is blocked in a low-level read, not executing R code. Two
## defenses, both applied here: (1) call this from a *non-interactive*
## session (`Rscript your_script.R` from Terminal, or `R -q -e '...'`) so
## any stray readline() errors out immediately ("no lines available in
## input connection") instead of blocking -- do NOT source() this
## interactively in RStudio for a long run; (2) the `timeout_sec` argument
## below, which best-effort-interrupts genuinely slow (not blocked)
## computations via setTimeLimit().
##
## @param link one or more names from link_functions (default: all four).
##   IMPORTANT: "linear" is the single *worst-case* link for showcasing
##   SIR-WLS/LassoSIR's advantage over plain LASSO -- under a truly linear
##   link, plain (penalized) linear regression is the *correctly specified*
##   model and is a hard benchmark to beat, whereas SIR-based methods pay a
##   price for not assuming linearity that isn't rewarded there. Their
##   advantage shows up under nonlinear links (especially "exponential" in
##   the full n=100,p=50/p=500 study: SIR-WLS beats LASSO by +0.24 to +0.42
##   in proximity there, vs. LASSO leading under "linear"). A run using only
##   the default `link = "linear"` of an earlier version of this function
##   is what produced the apparently-backwards SIR-WLS-loses-to-LASSO
##   pattern -- not a bug, just the least favorable link for SIR-WLS,
##   tested in isolation. Running all four (now the default) avoids that
##   trap and lets you compare like with like against Table~sim-link /
##   Table~sim-p500-link in the paper.
## @param mc.cores number of cores to use via parallel::mclapply() (forking,
##   so Mac/Linux only -- automatically falls back to serial on Windows or
##   if the `parallel` package is unavailable). Defaults to all-but-one
##   detected core.
run_p_scaling <- function(n = 100, p_grid = c(20, 50, 100, 300),
                           q = 5, repetitions = 5,
                           link = names(link_functions),
                           design = "autoR", rho = 0.5, nslice = 10,
                           timeout_sec = 30,
                           mc.cores = max(1, parallel::detectCores() - 1),
                           verbose = TRUE) {
  methods <- c("SIR", "SAVE", "LassoSIR", "SIR-NCT", "SIR-WLS")
  results <- expand.grid(p = p_grid, link = link, method = methods, stringsAsFactors = FALSE)
  results$abscos <- results$TPR <- results$FPR <- results$FDR <- results$Time <- NA_real_

  use_parallel <- .Platform$OS.type == "unix" && requireNamespace("parallel", quietly = TRUE) &&
    mc.cores > 1
  if (verbose) {
    message(if (use_parallel) paste0("Running in parallel on ", mc.cores, " cores.")
            else "Running serially (set mc.cores > 1 on Mac/Linux to parallelize).")
  }

  run_and_score <- function(Beta, betahat, time_sec) {
    betahat <- as.vector(betahat)
    c(abscos = abs(costeta(Beta, betahat)),
      TPR = TPR.funct(Beta, betahat),
      FPR = FPR.funct(Beta, betahat),
      FDR = fdr_funct(Beta, betahat),
      Time = time_sec)
  }

  ## safe_run(): tryCatch + setTimeLimit, and (if verbose) prints which
  ## method/p/replicate it is about to run *before* running it, so if a run
  ## does stall, the last printed line tells you exactly where.
  safe_run <- function(label, p, expr_fun) {
    if (verbose) message("  -> ", label, " (p = ", p, ") ...", appendLF = FALSE)
    t0 <- Sys.time()
    b <- tryCatch({
      setTimeLimit(elapsed = timeout_sec, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      expr_fun()
    }, error = function(e) {
      if (verbose) message(" FAILED/TIMED OUT (", conditionMessage(e), ")")
      rep(0, p)
    })
    dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (verbose && dt < timeout_sec) message(" ok (", round(dt, 2), "s)")
    list(b = b, dt = dt)
  }

  one_replicate <- function(p, Beta, link_fn) {
    X <- do.call(X_rand, list(n = n, p = p, q = q, S = seq_len(q),
                               type = design, rho = rho))
    t_index <- X %*% Beta
    eps <- rnorm(n)
    y <- as.vector(link_fn(t_index, eps))

    out <- list()
    r <- safe_run("SIR",      p, function() run_sir_classic(X, y));      out$SIR      <- run_and_score(Beta, r$b, r$dt)
    r <- safe_run("SAVE",     p, function() run_save_classic(X, y));     out$SAVE     <- run_and_score(Beta, r$b, r$dt)
    r <- safe_run("LassoSIR", p, function() run_lassosir(X, y, H = nslice)); out$LassoSIR <- run_and_score(Beta, r$b, r$dt)
    r <- safe_run("SIR-NCT",  p, function() run_nctsir(X, y, cv = TRUE)); out$`SIR-NCT` <- run_and_score(Beta, r$b, r$dt)
    r <- safe_run("SIR-WLS",  p, function() run_wlssir(X, y));           out$`SIR-WLS` <- run_and_score(Beta, r$b, r$dt)
    out
  }

  for (link_name in link) {
    link_fn <- link_functions[[link_name]]

    for (p in p_grid) {
      Beta <- c(rep(1, q), rep(0, p - q))
      if (verbose) message("=== link = ", link_name, ", p = ", p, " (", repetitions, " replications) ===")

      acc <- if (use_parallel) {
        parallel::mclapply(seq_len(repetitions), function(i) one_replicate(p, Beta, link_fn),
                            mc.cores = mc.cores)
      } else {
        lapply(seq_len(repetitions), function(i) one_replicate(p, Beta, link_fn))
      }

      for (m in methods) {
        vals <- t(sapply(acc, function(a) a[[m]]))
        means <- colMeans(vals, na.rm = TRUE)
        rows <- which(results$p == p & results$link == link_name & results$method == m)
        results[rows, c("abscos", "TPR", "FPR", "FDR", "Time")] <- as.list(means)
      }
      message("link = ", link_name, ", p = ", p, " done")
    }
  }

  results
}

## ---- 7. Dimension-estimation sub-study --------------------------------------
##
## @param ladle_B number of bootstrap resamples used by ladle_estimator().
##   The default (100) is fine at moderate p (e.g. p = 50), but each
##   resample re-does a p x p eigendecomposition, so cost scales roughly
##   like O(ladle_B * p^3); for a p >> n illustration (e.g. p = 300) it is
##   worth lowering this (e.g. 50) to keep runtime reasonable.
## @param timeout_sec per-criterion (ladle, chi2), per-replicate wall-clock
##   cap, exactly as in run_p_scaling() -- important once p >> n, where
##   Sigma_X is severely rank-deficient and eigen() on solve(Sigma + tiny
##   ridge) %*% Lambda can occasionally be slow or numerically degenerate;
##   a failed/timed-out call is recorded as NA rather than allowed to hang
##   the whole loop.
## @param verbose if TRUE, prints one line per (true_d, repetition) with
##   elapsed time, so a slow p >> n run shows visible progress instead of
##   producing no console output for minutes.
run_dimension_substudy <- function(n = 300, p = 100, repetitions = 30,
                                    true_d_values = 1:3, ladle_B = 100,
                                    timeout_sec = 60, verbose = TRUE) {

  out <- expand.grid(true_d = true_d_values, repetition = seq_len(repetitions))
  out$ladle_dhat <- out$chi2_dhat <- NA_integer_

  ## Wraps one criterion call in a wall-clock timeout, mirroring
  ## run_p_scaling()'s safe_run(); a fresh call/exit scope per invocation
  ## keeps the on.exit() timeout reset local to that one call.
  safe_call <- function(expr_fun) {
    tryCatch({
      setTimeLimit(elapsed = timeout_sec, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      expr_fun()
    }, error = function(e) list(d_hat = NA_integer_, error = conditionMessage(e)))
  }

  for (i in seq_len(nrow(out))) {
    d <- out$true_d[i]
    if (verbose) message("-> true_d = ", d, ", repetition ", out$repetition[i], "/", repetitions,
                          " (n = ", n, ", p = ", p, ") ...", appendLF = FALSE)
    t0 <- Sys.time()
    Beta_mat <- matrix(0, p, d)
    active <- sample(seq_len(p), 5 * d)
    for (j in seq_len(d)) Beta_mat[active[((j - 1) * 5 + 1):(j * 5)], j] <- 1

    X <- mvtnorm::rmvnorm(n, rep(0, p), diag(p))
    t_index <- X %*% Beta_mat
    eps <- rnorm(n)
    ## IMPORTANT (two rounds of DGP fixes, both needed):
    ## (1) y must depend on the d index directions through a genuinely
    ##     non-additive-linear link, otherwise the "true" central subspace
    ##     collapses to rank 1 regardless of d (`y <- rowSums(t_index) +
    ##     eps` is equivalent to a single linear index).
    ## (2) EVERY per-direction transform must be visible to first-moment
    ##     SIR, i.e. NOT symmetric about 0 -- Section~\ref{sec:link-function}
    ##     of the paper already notes that first-moment SIR is blind to any
    ##     direction along which E[X|Y] is flat, which includes symmetric
    ##     (even) link functions. The previous transform list used `t^2 -
    ##     1` for the second direction, an even function -- invisible to
    ##     SIR by construction, however large its coefficient -- which is
    ##     the most likely explanation for ladle_dhat being stuck below the
    ##     true d for d = 2, 3 even after fix (1) alone. All three
    ##     transforms below are odd/monotonic (identity, cube, sign-times-
    ##     sqrt) so none of the d directions is structurally invisible to
    ##     SIR-based eigenvalues, while remaining non-linearly related so
    ##     they cannot be summed away into a single linear index.
    transforms <- list(function(t) t,
                        function(t) t^3 / 3,
                        function(t) sign(t) * sqrt(abs(t)))
    y <- eps
    for (j in seq_len(d)) y <- y + transforms[[((j - 1) %% 3) + 1]](t_index[, j])
    y <- as.vector(y)

    ladle_out <- safe_call(function() ladle_estimator(X, y, Kmax = 8, B = ladle_B))
    out$ladle_dhat[i] <- ladle_out$d_hat

    ## Sequential chi-squared test (Li, 1991; Schott, 1994) -- see
    ## sir_chi2_dhat() in Section 5 for the criterion and for why it
    ## replaces the earlier ad hoc "BIC-type" criterion used here.
    chi2_out <- safe_call(function() sir_chi2_dhat(X, y, nslice = 10, alpha = 0.05))
    out$chi2_dhat[i] <- chi2_out$d_hat

    if (verbose) message(sprintf(" done (%.1fs)%s%s",
                                  as.numeric(difftime(Sys.time(), t0, units = "secs")),
                                  if (!is.null(ladle_out$error)) paste0(" [ladle: ", ladle_out$error, "]") else "",
                                  if (!is.null(chi2_out$error)) paste0(" [chi2: ", chi2_out$error, "]") else ""))
  }

  out
}

## ---- 7b. Diagnostic helper for the dimension sub-study ----------------------
##
## Kept as a diagnostic tool even after the two DGP/criterion fixes below,
## since the ladle estimator's remaining behaviour at true_d = 2, 3 is not
## yet fully understood (see the note at the bottom of this function). This
## function regenerates ONE replicate with the exact DGP now used by
## run_dimension_substudy() and prints every intermediate quantity that
## feeds both criteria.
##
## Two things changed here relative to the version you last ran
## (n = 150, p = 50, repetitions = 5: ladle_dhat = 1.0 / 1.6 / 0.0 and
## chi2_dhat -- then still labelled bic_dhat -- stuck at 8 for every
## true_d):
##   (1) The second per-direction transform was `t^2 - 1`, an EVEN
##       function -- and Section~\ref{sec:link-function} of the paper
##       already notes that first-moment SIR is blind to any direction
##       along which E[X|Y] is flat, which includes symmetric link
##       functions. That direction was therefore structurally invisible
##       to every method tested here, not just under-detected -- the most
##       likely explanation for the pattern you saw (true_d = 1 essentially
##       perfect; true_d = 2 a mix of 1s and 2s; true_d = 3 collapsing
##       toward 0, since 2 of its 3 directions were partly or fully SIR-
##       invisible). It has been replaced with `t^3 / 3`, odd like the
##       other two transforms.
##   (2) The "BIC-type" criterion -- which, as you correctly pointed out,
##       borrowed its name from wls.sir()'s predictor-*screening* BIC
##       rather than any actual dimension-estimation criterion, and whose
##       -log(cumulative ratio) term diverges toward the boundary by
##       construction -- has been replaced by the sequential chi-squared
##       test of Li (1991) / Schott (1994) that Section~\ref{sec:order-
##       determination} of the paper already cites as the classical
##       approach to this exact problem (see sir_chi2_dhat() in Section 5).
##
## If ladle_dhat is still off for true_d = 2, 3 once you re-run with these
## fixes, the printed f_n/phi_n/g_n vectors below remain the place to look
## next: in particular, whether phi_n(k) rises sharply as soon as k exceeds
## the number of directions SIR can actually see (rather than at true_d
## itself), which would point to a remaining eigenvector-instability issue
## in the bootstrap step rather than the DGP.
##
## @param true_d the structural dimension to simulate (matches
##   run_dimension_substudy()'s true_d_values)
## @param n, p sample size / ambient dimension (match the run_dimension_substudy()
##   call you are debugging, e.g. n = 150, p = 50)
## @param seed optional RNG seed for reproducibility across repeated calls
debug_dimension_estimation <- function(true_d, n = 300, p = 100, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  d <- true_d

  Beta_mat <- matrix(0, p, d)
  active <- sample(seq_len(p), 5 * d)
  for (j in seq_len(d)) Beta_mat[active[((j - 1) * 5 + 1):(j * 5)], j] <- 1

  X <- mvtnorm::rmvnorm(n, rep(0, p), diag(p))
  t_index <- X %*% Beta_mat
  eps <- rnorm(n)
  transforms <- list(function(t) t,
                      function(t) t^3 / 3,
                      function(t) sign(t) * sqrt(abs(t)))
  y <- eps
  for (j in seq_len(d)) y <- y + transforms[[((j - 1) %% 3) + 1]](t_index[, j])
  y <- as.vector(y)

  cat("==== debug_dimension_estimation: true_d =", d, ", n =", n, ", p =", p, "====\n\n")

  ## -- Ladle diagnostics -------------------------------------------------
  ladle_out <- tryCatch(ladle_estimator(X, y, Kmax = 8, B = 100),
                         error = function(e) { cat("ladle_estimator ERROR:", conditionMessage(e), "\n"); NULL })
  if (!is.null(ladle_out)) {
    cat("-- ladle_estimator() --\n")
    cat("d_hat =", ladle_out$d_hat, "\n")
    cat("eigenvalues (k = 0..8, i.e. lambda_1..lambda_9):\n")
    print(round(ladle_out$eigenvalues, 6))
    cat("f_n (normalized eigenvalue term, k = 0..8):\n")
    print(round(ladle_out$f_n, 6))
    cat("phi_raw (raw bootstrap subspace instability, k = 0..8):\n")
    print(round(ladle_out$phi_raw, 6))
    cat("phi_n (normalized bootstrap term, k = 0..8):\n")
    print(round(ladle_out$phi_n, 6))
    cat("g_n = f_n + phi_n (k = 0..8) -- argmin over this is d_hat:\n")
    print(round(ladle_out$g_n, 6))
    cat("\n")
  }

  ## -- Sequential chi-squared test diagnostics ----------------------------
  cat("-- sequential chi-squared test (Li 1991 / Schott 1994) --\n")
  chi2_out <- tryCatch(sir_chi2_dhat(X, y, nslice = 10, alpha = 0.05),
                        error = function(e) { cat("chi2 test ERROR:", conditionMessage(e), "\n"); NULL })
  if (!is.null(chi2_out)) {
    cat("d_hat =", chi2_out$d_hat, "\n")
    cat("test statistic Lambda_k = n * sum_{j>k} lambda_j, by k:\n")
    print(round(chi2_out$stat, 4))
    cat("df = (p-k)(H-1-k), by k:\n")
    print(chi2_out$df)
    cat("p-value, by k (first p-value > alpha = 0.05 stops the sequence):\n")
    print(round(chi2_out$pval, 4))
  }

  invisible(list(ladle = ladle_out, chi2 = chi2_out))
}

## ---- 8. Example run (small settings for a quick smoke test) ----------------
## Uncomment to run once your packages are loaded and source() above points
## to your copy of myfunctions.R. Start small (n, p, repetitions) to
## sanity-check timing before scaling up to the paper's final settings
## (e.g. n = 500, p = 700, repetitions = 200, matching the published
## SIR-WLS simulation scale).
##
## BEFORE scaling `repetitions` up from the pilot value (5) toward that
## final scale, run probe_simulation_time() at your target (n, p, q) --
## it runs a couple of real replications and extrapolates the time cost
## for larger repetitions counts, e.g.:
##   probe_simulation_time(n = 500, p = 700, q = 10)
## This tells you, from an actual timed run rather than a guess, whether
## repetitions = 100 or 200 is realistic on your machine before you commit
## to it -- run_simulation() itself is now parallelized (mclapply, like
## run_p_scaling()) and per-method-call timeout-protected, so a long run
## should no longer hang silently, but it can still legitimately take a
## long time at large (n, p, repetitions) if left uncapped.

 #res_main <- run_simulation(n = 100, p = 500, q = 5, repetitions = 100)
 #write.csv(res_main, "sir_review_simulation_results.csv", row.names = FALSE)
 #print(res_main)
#
 #res_dim <- run_dimension_substudy(n = 150, p = 50, repetitions = 100)
 #write.csv(res_dim, "sir_review_dimension_estimation_results.csv", row.names = FALSE)
 #print(aggregate(cbind(ladle_dhat, chi2_dhat) ~ true_d, data = res_dim, FUN = mean))
#
## p >> n companion run: n = 150, p = 500, the same high-dimensional point
## used in run_simulation()/run_p_scaling() above, to show directly (not
## just argue) that both order-determination criteria break down further
## once Sigma_X is severely rank-deficient, rather than just under-estimating
## d as at the moderate n = 150, p = 50 point above. ladle_B is lowered from
## the 100 default (each bootstrap resample costs a full p x p
## eigendecomposition, so cost scales with ladle_B * p^3). This function is
## a plain sequential loop (not parallelized like run_simulation()/
## run_p_scaling()), and near-singular Sigma_X can make individual
## replicates take several minutes each at this scale, so expect this call
## to run for hours at repetitions = 100.
# res_dim_highdim <- run_dimension_substudy(n = 150, p = 500, repetitions = 100,
                                          #  true_d_values = 1:3, ladle_B = 50)
# write.csv(res_dim_highdim, "sir_review_dimension_estimation_highdim_results.csv", row.names = FALSE)
# print(aggregate(cbind(ladle_dhat, chi2_dhat) ~ true_d, data = res_dim_highdim, FUN = mean))

## ---- 9. Plots -----------------------------------------------------------------
##
## These three functions reproduce, from your own res_main / res_dim data
## frames, the exact figure styles already embedded in the LaTeX paper
## (figures/fig_sim_link.pdf, figures/fig_sim_design.pdf,
## figures/fig_sim_tradeoff.pdf -- currently pilot-scale placeholders built
## from a 5-replication run). Re-running these on your final-scale results
## (e.g. n = 500, p = 700, repetitions = 200) and overwriting the same
## filenames under figures/ updates the paper's figures with no LaTeX
## changes needed. A fourth function, plot_dimension_substudy(), covers the
## dimension-estimation sub-study once you have re-run it with the current
## Section 7 fixes (odd/monotonic per-direction transforms; the sequential
## chi-squared test in place of the earlier ad hoc BIC).
##
## All four save a PDF via ggsave() and also return the ggplot object
## invisibly, so you can inspect/tweak it interactively before saving.

method_levels <- c("SIR", "SAVE", "LassoSIR", "SIR-NCT", "SIR-WLS")
method_palette <- c(SIR = "#1b4965", SAVE = "#5fa8d3", LassoSIR = "#bc4749",
                     `SIR-NCT` = "#e09f3e", `SIR-WLS` = "#588157")

## Proximity |cos(beta_hat, beta)| against link function, one curve per
## method, averaged over design -- reproduces Figure "fig:sim-link".
plot_link_comparison <- function(res_main,
                                  link_levels = c("linear", "cubic", "exponential", "sine"),
                                  file = "figures/fig_sim_link.pdf") {
  agg <- aggregate(abscos ~ method + link, data = res_main, FUN = mean)
  agg$method <- factor(agg$method, levels = method_levels)
  agg$link <- factor(agg$link, levels = link_levels)

  p <- ggplot(agg, aes(x = link, y = abscos, group = method, color = method)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.2) +
    scale_color_manual(values = method_palette) +
    scale_x_discrete(labels = tools::toTitleCase(link_levels)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = "Link function", y = expression("Proximity " * "|" * cos(hat(beta), beta) * "|"),
         color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())

  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  ggsave(file, p, width = 5.6, height = 3.8)
  print(p)      # forces the plot to also show in R's plot pane, not just save to disk
  invisible(p)
}

## Proximity |cos(beta_hat, beta)| against design, one curve per method,
## averaged over link -- reproduces Figure "fig:sim-design".
plot_design_comparison <- function(res_main,
                                    design_levels = c("block", "autoR", "homog2"),
                                    design_labels = c("Block", "AutoR", "Homog2"),
                                    file = "figures/fig_sim_design.pdf") {
  agg <- aggregate(abscos ~ method + design, data = res_main, FUN = mean)
  agg$method <- factor(agg$method, levels = method_levels)
  agg$design <- factor(agg$design, levels = design_levels, labels = design_labels)

  p <- ggplot(agg, aes(x = design, y = abscos, group = method, color = method)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.2) +
    scale_color_manual(values = method_palette) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = "Design", y = expression("Proximity " * "|" * cos(hat(beta), beta) * "|"),
         color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())

  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  ggsave(file, p, width = 5.0, height = 3.8)
  print(p)      # forces the plot to also show in R's plot pane, not just save to disk
  invisible(p)
}

## TPR-vs-FDR trade-off, overall averages across all designs/links, point
## size proportional to mean computation time -- reproduces Figure
## "fig:sim-tradeoff".
plot_tradeoff <- function(res_main, file = "figures/fig_sim_tradeoff.pdf") {
  agg <- aggregate(cbind(TPR, FDR, Time) ~ method, data = res_main, FUN = mean)
  agg$method <- factor(agg$method, levels = method_levels)

  p <- ggplot(agg, aes(x = FDR, y = TPR, color = method, size = Time, label = method)) +
    geom_point(alpha = 0.85) +
    ggrepel_or_text(agg) +
    scale_color_manual(values = method_palette) +
    scale_size_continuous(range = c(3, 10), guide = "none") +
    coord_cartesian(xlim = c(0, 1), ylim = c(min(0.55, min(agg$TPR) - 0.05), 1.05)) +
    labs(x = "False Discovery Rate (FDR)", y = "True Positive Rate (TPR)", color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none", panel.grid.minor = element_blank())

  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  ggsave(file, p, width = 5.2, height = 4.2)
  print(p)      # forces the plot to also show in R's plot pane, not just save to disk
  invisible(p)
}

## Performance against p at fixed n, one curve per method, with a dashed
## reference line at p = n -- reproduces run_p_scaling() (Section 6b). This
## is the one figure in this script that is NOT a re-plot of a results
## table already in the paper: it directly shows the p<n / p~=n / p>n
## breakdown behaviour discussed in Section 3.2.
##
## @param metric one of "abscos", "TPR", "FPR", "FDR" (a column of res_scaling)
## @param link optional: restrict to one link (e.g. "exponential") before
##   plotting. If NULL (default) and res_scaling contains more than one
##   link, all of them are shown as separate facets (facet_wrap(~link)) --
##   do NOT average across links here, since "linear" specifically is the
##   single worst-case link for SIR-WLS/LassoSIR (see the long comment
##   above run_p_scaling()); collapsing links together would hide exactly
##   the pattern that matters.
plot_p_scaling <- function(res_scaling, metric = "abscos", n_ref = NULL, link = NULL,
                            file = paste0("figures/fig_p_scaling_", metric,
                                          if (!is.null(link)) paste0("_", link) else "", ".pdf")) {
  if (!is.null(link)) res_scaling <- res_scaling[res_scaling$link == link, ]
  facet_by_link <- is.null(link) && "link" %in% names(res_scaling) &&
    length(unique(res_scaling$link)) > 1

  form <- if (facet_by_link) paste(metric, "~ method + p + link") else paste(metric, "~ method + p")
  agg <- aggregate(as.formula(form), data = res_scaling, FUN = mean)
  agg$method <- factor(agg$method, levels = method_levels)
  if (facet_by_link) agg$link <- tools::toTitleCase(agg$link)

  ylab <- switch(metric,
                  abscos = expression("Proximity " * "|" * cos(hat(beta), beta) * "|"),
                  TPR = "True Positive Rate",
                  FPR = "False Positive Rate",
                  FDR = "False Discovery Rate",
                  metric)

  p_plot <- ggplot(agg, aes(x = p, y = .data[[metric]], color = method, group = method))
  if (!is.null(n_ref)) {
    p_plot <- p_plot +
      geom_vline(xintercept = n_ref, linetype = "dashed", color = "grey50") +
      annotate("text", x = n_ref, y = Inf, label = "p = n", vjust = 1.5, hjust = -0.1,
                size = 3.2, color = "grey40")
  }
  p_plot <- p_plot +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.2) +
    scale_color_manual(values = method_palette) +
    labs(x = "Number of predictors p", y = ylab, color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())
  if (facet_by_link) p_plot <- p_plot + facet_wrap(~link, nrow = 1)

  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  ggsave(file, p_plot, width = if (facet_by_link) 10.5 else 5.8, height = 3.8)
  print(p_plot)   # forces the plot to also show in R's plot pane, not just save to disk
  invisible(p_plot)
}

## Small helper: use ggrepel if installed (nicer non-overlapping labels),
## otherwise fall back to plain geom_text.
ggrepel_or_text <- function(agg) {
  if (requireNamespace("ggrepel", quietly = TRUE)) {
    ggrepel::geom_text_repel(data = agg, size = 3.2, show.legend = FALSE)
  } else {
    geom_text(data = agg, size = 3.2, vjust = -1.1, show.legend = FALSE)
  }
}

## Dimension-estimation sub-study: mean (+/- 1 SE) of ladle_dhat and
## chi2_dhat against the true structural dimension d, once you have re-run
## run_dimension_substudy() with the current Section 7 fixes applied. Not
## yet embedded in the LaTeX paper (Section~\ref{sec:simulation-pscaling}
## notes that a corrected run is still pending).
plot_dimension_substudy <- function(res_dim, file = "figures/fig_dim_substudy.pdf") {
  long <- reshape(res_dim, direction = "long", varying = c("ladle_dhat", "chi2_dhat"),
                   v.names = "dhat", timevar = "criterion",
                   times = c("Ladle", "Chi-squared"), idvar = c("true_d", "repetition"))
  summ <- aggregate(dhat ~ true_d + criterion, data = long,
                     FUN = function(v) c(mean = mean(v, na.rm = TRUE),
                                          se = sd(v, na.rm = TRUE) / sqrt(sum(!is.na(v)))))
  summ <- do.call(data.frame, summ)
  names(summ)[3:4] <- c("mean_dhat", "se_dhat")

  p <- ggplot(summ, aes(x = true_d, y = mean_dhat, color = criterion, group = criterion)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.2) +
    geom_errorbar(aes(ymin = mean_dhat - se_dhat, ymax = mean_dhat + se_dhat), width = 0.08) +
    scale_x_continuous(breaks = sort(unique(summ$true_d))) +
    labs(x = "True structural dimension d", y = expression(hat(d) * " (mean " %+-% " 1 SE)"),
         color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())

  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  ggsave(file, p, width = 5.0, height = 3.8)
  print(p)      # forces the plot to also show in R's plot pane, not just save to disk
  invisible(p)
}

## Example: res_main / res_dim (smoke-test scale) -- still commented out,
## since re-running these every time you source() the file is slow and you
## already have a validated res_main from the earlier pilot run. Uncomment
## if you want to regenerate them.
# plot_link_comparison(res_main)
# plot_design_comparison(res_main)
# plot_tradeoff(res_main)
# plot_dimension_substudy(res_dim)

## p-scaling run: UNCOMMENTED so that sourcing this file (or running it with
## Rscript -- recommended, see the long comment above run_p_scaling() for
## why) directly executes the pilot p-scaling study and writes the plots.
## Defaults now include p = 300 with n = 100, i.e. p = 3n, a genuinely
## high-dimensional (p > n) point alongside p = 20 (p << n), p = 50 (p < n),
## and p = 100 (p = n, the exact boundary case) -- so the resulting curve
## shows the full p<n / p=n / p>n crossing rather than stopping at the
## boundary. `link` is left at its default (all four link functions, looped
## automatically) rather than pinned to "linear" -- "linear" alone is the
## single worst-case link for SIR-WLS/LassoSIR (see the long comment above
## run_p_scaling()), and an earlier version of this example that hard-coded
## link = "linear" is what produced the apparently-backwards
## SIR-WLS-loses-to-LASSO pattern. This means the run now takes ~4x as long
## (4 links x 4 p-values x 6 methods); reduce p_grid or repetitions further
## if needed, or pass a single link explicitly to iterate faster.
res_scaling <- run_p_scaling(n = 100, p_grid = c(20, 50, 100, 300), q = 5, repetitions = 5)
write.csv(res_scaling, "sir_review_p_scaling_results.csv", row.names = FALSE)
print(res_scaling)

## Once this small run finishes cleanly, scale up gradually (more
## repetitions, finer grid) rather than jumping straight to very large p:
# res_scaling <- run_p_scaling(n = 150, p_grid = c(20, 50, 100, 150, 250, 450),
#                               repetitions = 10, mc.cores = parallel::detectCores() - 1)

## With 4 links in res_scaling, these now render as 4 side-by-side facets
## (one per link) rather than a single curve -- compare the "linear" facet
## against "exponential" to see the SIR-WLS/LASSO ranking flip directly.
plot_p_scaling(res_scaling, metric = "abscos", n_ref = 100)
plot_p_scaling(res_scaling, metric = "TPR",    n_ref = 100)
plot_p_scaling(res_scaling, metric = "FDR",    n_ref = 100)
