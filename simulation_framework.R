## =============================================================================
## simulation_framework.R
##
## Comparative simulation study for:
##   "Sliced Inverse Regression Revisited: A Comprehensive Review with
##    Simulation and Real Data Applications" (Asrir)
##
## Purpose
## -------
## A systematic 3-factor simulation study comparing five sufficient
## dimension reduction estimators -- SIR, SAVE, LassoSIR, SIR-NCT, and
## SIR-WLS -- across:
##   (1) sensitivity to the link function (linear / cubic / exponential /
##       non-monotone sine),
##   (2) three correlation designs (block-exchangeable, autoregressive, and
##       a sparse-signal design, "homog2", in which the active and inactive
##       predictor blocks are each internally equicorrelated but only
##       weakly cross-correlated), and
##   (3) reliability of structural dimension (d) estimation, compared
##       between a ladle-estimator implementation and the sequential
##       chi-squared test of Li (1991) / Schott (1994).
##
## Estimator and metric implementation notes
## ------------------------------------------
##   - LassoSIR is called directly from the installed `LassoSIR` package:
##     `LassoSIR(X, y, H, no.dim = 1, solution.path = FALSE,
##     categorical = FALSE, nfolds = 10, screening = FALSE)`, which returns
##     a list with a `$beta` matrix.
##   - FDR is computed by a local `fdr_funct()` defined below (the
##     FP/(FP+TP) definition), consistent with the TPR/FPR metrics used
##     throughout.
##   - `X_rand()` implements `type` %in% c("block","autoR","homog","homog2");
##     the design grid below uses "homog2" (the sparse-signal correlation
##     structure of Lin, Zhao & Liu, 2019), passing its `S` argument
##     explicitly as `S = seq_len(q)`.
##   - `wls.sir()` returns a list with `$betahat`; `sir.nct()` returns the
##     beta vector directly.
##
## Requires the `sirwls` and `nctsir` packages (see below), together with
## `dr`, `Rdimtools`, `glmnet`, and `mvtnorm`.
##
## =============================================================================

## ---- 0. Packages ---------------------------------------------------------------

## Install once if needed:
# install.packages(c("dr", "Rdimtools", "glmnet", "mvtnorm", "LassoSIR",
#                     "ggplot2", "devtools"))
# devtools::install_github("nasrir-ai/SIR-WLS")
# devtools::install_github("nasrir-ai/nctsir", subdir = "nctsir 2")

library(dr)
library(Rdimtools)
library(glmnet)
library(mvtnorm)
library(LassoSIR)   # provides LassoSIR() directly -- see run_lassosir() below
library(ggplot2)    # only needed for Section 9 (plots)

## Estimator/utility functions this script depends on -- wls.sir(), X_rand(),
## TPR.funct(), FPR.funct(), costeta() -- are provided by the sirwls package
## (companion to Asrir & Mkhadri, 2025, JSCS), and sir.nct() by the nctsir
## package (companion to Asrir & Mkhadri, 2025, CSTM).
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

## ---- 2. Design factor ---------------------------------------------------------
##
## X_rand()'s `type` argument supports "block", "autoR", "homog", and
## "homog2". "homog2" is the sparse-signal correlation structure of Lin,
## Zhao & Liu (2019): predictors
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
## Zhao & Liu, 2019). Returns a list with a `$beta` matrix.
run_lassosir <- function(X, y, H = 10, ndim = 1) {
  fit <- LassoSIR(X, y, H, no.dim = ndim, solution.path = FALSE,
                   categorical = FALSE, nfolds = 10, screening = FALSE)
  as.matrix(fit$beta)
}

## Nadaraya-Watson kernel regression with a Gaussian kernel, used below as
## the out-of-fold prediction step in cv_select_nctsir(). Self-contained
## (base R only, no extra package).
nw <- function(x, X, Y, h) {
  sapply(x, function(x0) {
    w <- dnorm((X - x0) / h)
    if (sum(w) == 0) return(mean(Y))
    sum(w * Y) / sum(w)
  })
}

## ---- 3b. Cross-validated (K, tau) selection for SIR-NCT ---------------------
##
## SIR-NCT's first-stage compression dimension K and soft-threshold level
## tau are both cross-validated rather than fixed, because a fixed K that
## saturates at n (i.e. K = min(n, p) once p >= n) yields no actual
## compression relative to the sample size, causing SIR-NCT's first-stage
## SIR fit to inherit classical SIR's own p ~ n breakdown. cv_select_nctsir()
## cross-validates both K and tau jointly, with every candidate K kept
## strictly below n by construction, using Nadaraya-Watson kernel
## regression (nw()) for the out-of-fold prediction step.
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

## sir.nct(): signature sir.nct(x, y, k, tho, threshold = "soft");
## returns the beta-hat vector directly (ndim is implicitly 1 -- do.sir()
## is called internally with ndim = 1).
##
## @param cv if TRUE (the default), or if k/tau are left NULL, K and tau are
##   selected via cv_select_nctsir() above instead of using the
##   caller-supplied fixed values, to avoid the K = min(n, p) trap
##   described above.
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
## costeta(), TPR.funct(), and FPR.funct() are provided by the sirwls
## package. fdr_funct() below computes the complementary FP/(FP+TP)
## definition on beta-vectors directly.
fdr_funct <- function(beta, hatbeta) {
  TP <- sum(beta != 0 & hatbeta != 0)
  FP <- sum(hatbeta != 0 & beta == 0)
  if ((FP + TP) == 0) return(0)
  FP / (FP + TP)
}

## ---- 5. SIR eigen-decomposition helper + ladle estimator (Luo & Li, 2016) ---
##
## sir_eigen(): shared helper computing the sample SIR generalized
## eigenvalues/eigenvectors of Sigma_X^{-1} Lambda directly, since
## `Rdimtools::do.sir()` only returns `$Y`, `$trfinfo`, and `$projection`
## (no eigenvalue field). This helper is used both by `ladle_estimator()`
## below and by `sir_chi2_dhat()`, so the two order-determination criteria
## are compared on eigenvalues computed the same way.
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
## Schott, 1994), used in run_dimension_substudy() as the second
## order-determination criterion, alongside the ladle estimator below.
##
## Under H0: d = k (only the first k population eigenvalues of
## Sigma_X^{-1} Lambda are non-zero),
##   Lambda_k = n * sum_{j=k+1}^{p} lambda_j  ~approx  chi^2_{(p-k)(H-1-k)}
## Testing sequentially from k = 0 upward and stopping at the first k that
## is not rejected gives d_hat.
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
## whose minimizer estimates the structural dimension d, following Luo &
## Li (2016).
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
## Repetitions are run in parallel (parallel::mclapply()), each method call
## is protected by a wall-clock timeout (setTimeLimit()), and progress is
## reported per (design, link) cell. probe_simulation_time() gives a timing
## estimate for a given (n, p) before launching a long run.
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

## Timing probe: runs run_simulation() with a small number of repetitions
## (default 2) at a given (n, p, q) and extrapolates the total running time
## for larger numbers of repetitions (approximate when mc.cores > 1).
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
## Fixes n and a single design/link and sweeps p over a grid crossing n
## (p < n, p ~ n, p > n), illustrating the phase-transition result of
## Lin, Zhao & Liu (2018) discussed in Section 3.2 (sqrt(n)-consistency
## requires p = o(n)).
##
## Every method call is wrapped in tryCatch() *and* a hard per-call wall-clock
## timeout (see `timeout_sec` below): once p >= n, Sigma_X is singular, and
## classical (non-regularized) SIR/SAVE/SIR-NCT are expected to fail or
## degrade sharply -- that failure is the point of the plot, not a bug to
## hide, so a failed or timed-out replicate is scored as a zero vector
## (worst case) rather than dropped or left to hang the whole run.
##
## For long runs, calling the function from a non-interactive session
## (Rscript) is recommended.
##
## @param link one or more names from link_functions (default: all four).
##   Using all four links allows a like-with-like reading of the link
##   comparison tables in the paper.
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

  ## safe_run(): tryCatch + setTimeLimit, with an optional progress message.
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
    ## Non-additive link with odd (monotone) transforms, so that each of the
    ## d directions is visible to first-moment SIR and the central subspace
    ## has dimension d (see Section~\ref{sec:link-function}).
    transforms <- list(function(t) t,
                        function(t) t^3 / 3,
                        function(t) sign(t) * sqrt(abs(t)))
    y <- eps
    for (j in seq_len(d)) y <- y + transforms[[((j - 1) %% 3) + 1]](t_index[, j])
    y <- as.vector(y)

    ladle_out <- safe_call(function() ladle_estimator(X, y, Kmax = 8, B = ladle_B))
    out$ladle_dhat[i] <- ladle_out$d_hat

    ## Sequential chi-squared test (Li, 1991; Schott, 1994) -- see
    ## sir_chi2_dhat() in Section 5.
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
## Regenerates one replicate with the exact DGP used by
## run_dimension_substudy() and prints every intermediate quantity feeding
## both order-determination criteria (ladle estimator and sequential
## chi-squared test). Per-direction transforms are all odd functions
## (`t`, `t^3/3`, ...), since a first-moment method like SIR is blind to
## any direction along which E[X|Y] is flat, which includes symmetric
## (even) link functions -- an even transform would make that direction
## structurally invisible rather than merely under-detected.
##
##
## @param true_d the structural dimension to simulate (matches
##   run_dimension_substudy()'s true_d_values)
## @param n, p sample size / ambient dimension (match the run_dimension_substudy()
##   call, e.g. n = 150, p = 50)
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
## Settings used in the paper are given below (commented out).
## probe_simulation_time() can be used first to estimate running time, e.g.
##   probe_simulation_time(n = 500, p = 700, q = 10)


 #res_main <- run_simulation(n = 100, p = 500, q = 5, repetitions = 100)
 #write.csv(res_main, "sir_review_simulation_results.csv", row.names = FALSE)
 #print(res_main)
#
 #res_dim <- run_dimension_substudy(n = 150, p = 50, repetitions = 100)
 #write.csv(res_dim, "sir_review_dimension_estimation_results.csv", row.names = FALSE)
 #print(aggregate(cbind(ladle_dhat, chi2_dhat) ~ true_d, data = res_dim, FUN = mean))
#
## High-dimensional setting (n = 150, p = 500) for the dimension sub-study.
## ladle_B is reduced to 50 because each bootstrap resample requires a
## p x p eigendecomposition; this run is computationally long.
# res_dim_highdim <- run_dimension_substudy(n = 150, p = 500, repetitions = 100,
                                          #  true_d_values = 1:3, ladle_B = 50)
# write.csv(res_dim_highdim, "sir_review_dimension_estimation_highdim_results.csv", row.names = FALSE)
# print(aggregate(cbind(ladle_dhat, chi2_dhat) ~ true_d, data = res_dim_highdim, FUN = mean))

## ---- 9. Plots -----------------------------------------------------------------
##
## Functions producing the figures of the paper (fig_sim_link.pdf,
## fig_sim_design.pdf, fig_sim_tradeoff.pdf), plus plot_dimension_substudy()
## for the dimension-estimation sub-study. Each saves a PDF via ggsave()
## and returns the ggplot object invisibly.

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
## reference line at p = n (Section 3.2).
##
## @param metric one of "abscos", "TPR", "FPR", "FDR" (a column of res_scaling)
## @param link optional: restrict to one link before plotting. If NULL and
##   several links are present, each link is shown in its own facet.
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
## chi2_dhat against the true structural dimension d, from
## run_dimension_substudy()'s output.
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

## Example (commented out):
# plot_link_comparison(res_main)
# plot_design_comparison(res_main)
# plot_tradeoff(res_main)
# plot_dimension_substudy(res_dim)

## p-scaling run: UNCOMMENTED so that sourcing this file (or running it with
## Rscript -- recommended, see the long comment above run_p_scaling() for
## why) directly executes the p-scaling study and writes the plots.
## p_grid spans p = 20 (p << n), p = 50 (p < n), p = 100 (p = n, the exact
## boundary case), and p = 300 (p = 3n, genuinely high-dimensional), so the
## resulting curve shows the full p<n / p=n / p>n crossing. `link` is left
## at its default (all four link functions, looped automatically), which
## is important since SIR-type methods' comparative behaviour differs
## substantially across links (see the long comment above
## run_p_scaling()). This means the run takes ~4x as long as a single-link
## run (4 links x 4 p-values x 5 methods); reduce p_grid or repetitions if
## needed, or pass a single link explicitly to iterate faster. The example
## below uses a small `repetitions` for a quick sanity check; scale up to
## the paper's final setting (repetitions = 100) once it runs cleanly.
res_scaling <- run_p_scaling(n = 100, p_grid = c(20, 50, 100, 300), q = 5, repetitions = 5)
write.csv(res_scaling, "sir_review_p_scaling_results.csv", row.names = FALSE)
print(res_scaling)

## Full-scale run matching the paper's reported results:
# res_scaling <- run_p_scaling(n = 100, p_grid = c(20, 50, 100, 300), q = 5,
#                               repetitions = 100)

## With 4 links in res_scaling, these render as 4 side-by-side facets (one
## per link) rather than a single curve.
plot_p_scaling(res_scaling, metric = "abscos", n_ref = 100)
plot_p_scaling(res_scaling, metric = "TPR",    n_ref = 100)
plot_p_scaling(res_scaling, metric = "FDR",    n_ref = 100)
