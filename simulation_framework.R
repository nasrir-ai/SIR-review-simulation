## =============================================================================
## simulation_framework.R
##
## Simulation study of the review paper
##   "Sliced Inverse Regression Revisited: A Comprehensive Review with
##    Simulation and Real Data Applications" (Asrir).
##
## Five estimators (SIR, SAVE, LassoSIR, SIR-NCT, SIR-WLS) are compared across
##   (1) four link functions (linear, cubic, exponential, sine),
##   (2) three correlation designs (block, autoregressive, and the
##       sparse-signal design "homog2" of Lin, Zhao and Liu, 2019),
##   (3) the number of predictors p (p < n, p = n, p > n),
## and two order-determination criteria for the structural dimension d
## (ladle estimator; sequential chi-squared test of Li, 1991, and
## Schott, 1994) are compared in a separate sub-study.
##
## Requires the packages dr, Rdimtools, glmnet, mvtnorm, LassoSIR, ggplot2,
## sirwls (SIR-WLS) and nctsir (SIR-NCT); see README.md.
## =============================================================================

## ---- 0. Packages ------------------------------------------------------------

## Installation (once):
# install.packages(c("dr", "Rdimtools", "glmnet", "mvtnorm", "LassoSIR",
#                    "ggplot2", "devtools"))
# devtools::install_github("nasrir-ai/SIR-WLS")
# devtools::install_github("nasrir-ai/nctsir", subdir = "nctsir 2")

library(dr)
library(Rdimtools)
library(glmnet)
library(mvtnorm)
library(LassoSIR)   # provides LassoSIR() directly -- see run_lassosir() below
library(ggplot2)    # only needed for Section 9 (plots)

## sirwls: wls.sir(), X_rand(), TPR.funct(), FPR.funct(), costeta()
## nctsir: sir.nct()
library(sirwls)
library(nctsir)
## ---- 1. Link functions --------------------------------------------------------

link_functions <- list(
  linear      = function(t, eps) t + eps,
  cubic       = function(t, eps) (t^3) / 2 + eps,
  exponential = function(t, eps) exp(t) + eps,
  sine        = function(t, eps) sin(pi * t / 2) + eps
)

## ---- 2. Correlation designs (argument `type` of X_rand()) --------------------
## "homog2": active and inactive predictors are each equicorrelated at rho,
## with cross-correlation 0.1 (Lin, Zhao and Liu, 2019).
design_grid <- list(
  block  = list(type = "block",  alpha = c(0.2, 0.3, 0.7)),
  autoR  = list(type = "autoR",  rho   = 0.5),
  homog2 = list(type = "homog2", rho   = 0.6)
)

## ---- 3. Estimators -----------------------------------------------------------
## Each wrapper returns beta-hat as a p x ndim matrix.

run_sir_classic <- function(X, y, ndim = 1) {
  out <- Rdimtools::do.sir(X, y, ndim = ndim)
  as.matrix(Re(out$projection))
}

run_save_classic <- function(X, y, ndim = 1) {
  out <- Rdimtools::do.save(X, y, ndim = ndim)
  as.matrix(Re(out$projection))
}

## LassoSIR (Lin, Zhao and Liu, 2019).
run_lassosir <- function(X, y, H = 10, ndim = 1) {
  fit <- LassoSIR(X, y, H, no.dim = ndim, solution.path = FALSE,
                   categorical = FALSE, nfolds = 10, screening = FALSE)
  as.matrix(fit$beta)
}

## Nadaraya-Watson regression with a Gaussian kernel (used by cv_select_nctsir()).
nw <- function(x, X, Y, h) {
  sapply(x, function(x0) {
    w <- dnorm((X - x0) / h)
    if (sum(w) == 0) return(mean(Y))
    sum(w * Y) / sum(w)
  })
}

## SIR-NCT: K and tau chosen jointly by 5-fold cross-validation, with
## K < n for every candidate; the prediction error is that of a
## Nadaraya-Watson fit of y on the estimated index.
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

## SIR-NCT: sir.nct(x, y, k, tho, threshold); K and tau by cross-validation if cv = TRUE.
run_nctsir <- function(X, y, k = NULL, tau = NULL, threshold = "soft",
                        cv = TRUE, ...) {
  if (cv || is.null(k) || is.null(tau)) {
    sel <- cv_select_nctsir(X, y, threshold = threshold, ...)
    k <- sel$K; tau <- sel$tau
  }
  as.matrix(sir.nct(X, y, k, tau, threshold = threshold))
}

## SIR-WLS: wls.sir() returns betahat, zero outside the retained set.
run_wlssir <- function(X, y, ndim = 1) {
  fit <- wls.sir(X, y, ndim = ndim)
  as.matrix(Re(fit$betahat))
}

## ---- 4. Scoring ------------------------------------------------------------
## costeta(), TPR.funct() and FPR.funct() come from sirwls; FDR = FP / (FP + TP).
fdr_funct <- function(beta, hatbeta) {
  TP <- sum(beta != 0 & hatbeta != 0)
  FP <- sum(hatbeta != 0 & beta == 0)
  if ((FP + TP) == 0) return(0)
  FP / (FP + TP)
}

## ---- 5. Order determination ---------------------------------------------------
## Eigenvalues and eigenvectors of Sigma_X^{-1} Lambda (SIR kernel matrix),
## shared by the two criteria below.
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

## Sequential chi-squared test (Li, 1991; Schott, 1994). Under H0: d = k,
##   n * sum_{j>k} lambda_j  ~  chi^2_{(p-k)(H-1-k)};
## d_hat is the first k that is not rejected.
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

## Ladle estimator (Luo and Li, 2016).
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
      # 1 - |det(V_k^T Vb_k)|
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

## ---- 6. Main simulation: design x link x method --------------------------------
## Each method call is limited to timeout_sec seconds; a failed call is scored
## as a zero vector. Replications run in parallel with parallel::mclapply()
## on Mac/Linux. With `seed` set, results are reproducible for a given
## number of cores.
run_simulation <- function(n = 300, p = 200, q = 10, repetitions = 50,
                            nslice = 10, timeout_sec = 120,
                            mc.cores = max(1, parallel::detectCores() - 1),
                            verbose = TRUE, seed = 1) {
  if (!is.null(seed)) { RNGkind("L'Ecuyer-CMRG"); set.seed(seed) }

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
    ## S is used by X_rand() for type = "homog2" only
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

## Timing probe: runs run_simulation() with a few replications and extrapolates.
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

## ---- 6b. Performance as a function of p ----------------------------------------
## n fixed, p crossing n (p < n, p = n, p > n), for each link function.
run_p_scaling <- function(n = 100, p_grid = c(20, 50, 100, 300),
                           q = 5, repetitions = 5,
                           link = names(link_functions),
                           design = "autoR", rho = 0.5, nslice = 10,
                           timeout_sec = 30,
                           mc.cores = max(1, parallel::detectCores() - 1),
                           verbose = TRUE, seed = 1) {
  if (!is.null(seed)) { RNGkind("L'Ecuyer-CMRG"); set.seed(seed) }
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

## ---- 7. Dimension-estimation sub-study ------------------------------------------
## ladle_B: bootstrap resamples of the ladle estimator (cost ~ ladle_B * p^3).
## A failed or timed-out criterion is recorded as NA.
run_dimension_substudy <- function(n = 300, p = 100, repetitions = 30,
                                    true_d_values = 1:3, ladle_B = 100,
                                    timeout_sec = 60, verbose = TRUE, seed = 1) {
  if (!is.null(seed)) set.seed(seed)

  out <- expand.grid(true_d = true_d_values, repetition = seq_len(repetitions))
  out$ladle_dhat <- out$chi2_dhat <- NA_integer_

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
    ## odd (non-symmetric) transforms, so that every direction is visible to
    ## first-moment SIR and the d directions cannot be summed into one
    transforms <- list(function(t) t,
                        function(t) t^3 / 3,
                        function(t) sign(t) * sqrt(abs(t)))
    y <- eps
    for (j in seq_len(d)) y <- y + transforms[[((j - 1) %% 3) + 1]](t_index[, j])
    y <- as.vector(y)

    ladle_out <- safe_call(function() ladle_estimator(X, y, Kmax = 8, B = ladle_B))
    out$ladle_dhat[i] <- ladle_out$d_hat

    ## sequential chi-squared test
    chi2_out <- safe_call(function() sir_chi2_dhat(X, y, nslice = 10, alpha = 0.05))
    out$chi2_dhat[i] <- chi2_out$d_hat

    if (verbose) message(sprintf(" done (%.1fs)%s%s",
                                  as.numeric(difftime(Sys.time(), t0, units = "secs")),
                                  if (!is.null(ladle_out$error)) paste0(" [ladle: ", ladle_out$error, "]") else "",
                                  if (!is.null(chi2_out$error)) paste0(" [chi2: ", chi2_out$error, "]") else ""))
  }

  out
}

## ---- 8. Runs reported in the paper --------------------------------------------
## Run from a terminal with Rscript for long runs.
# res_main <- run_simulation(n = 100, p = 500, q = 5, repetitions = 100)
# write.csv(res_main, "sir_review_simulation_results.csv", row.names = FALSE)
#
# res_dim <- run_dimension_substudy(n = 150, p = 50, repetitions = 100)
# write.csv(res_dim, "sir_review_dimension_estimation_results.csv", row.names = FALSE)
#
# res_dim_highdim <- run_dimension_substudy(n = 150, p = 500, repetitions = 100,
#                                           true_d_values = 1:3, ladle_B = 50)
# write.csv(res_dim_highdim, "sir_review_dimension_estimation_highdim_results.csv",
#           row.names = FALSE)
#
# res_scaling <- run_p_scaling(n = 100, p_grid = c(20, 50, 100, 300), q = 5,
#                              repetitions = 100)
# write.csv(res_scaling, "sir_review_p_scaling_results.csv", row.names = FALSE)

## ---- 9. Figures -----------------------------------------------------------------
## Each function saves a PDF under figures/ and returns the ggplot object.

method_levels <- c("SIR", "SAVE", "LassoSIR", "SIR-NCT", "SIR-WLS")
method_palette <- c(SIR = "#1b4965", SAVE = "#5fa8d3", LassoSIR = "#bc4749",
                     `SIR-NCT` = "#e09f3e", `SIR-WLS` = "#588157")

## Proximity against the link function, averaged over designs.
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
  print(p)
  invisible(p)
}

## Proximity against the design, averaged over link functions.
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
  print(p)
  invisible(p)
}

## TPR against FDR, averaged over designs and links; point size = mean time.
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
  print(p)
  invisible(p)
}

## Performance against p at fixed n, one panel per link, dashed line at p = n.
## metric: "abscos", "TPR", "FPR" or "FDR".
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
  print(p_plot)
  invisible(p_plot)
}

## Labels with ggrepel if installed, plain text otherwise.
ggrepel_or_text <- function(agg) {
  if (requireNamespace("ggrepel", quietly = TRUE)) {
    ggrepel::geom_text_repel(data = agg, size = 3.2, show.legend = FALSE)
  } else {
    geom_text(data = agg, size = 3.2, vjust = -1.1, show.legend = FALSE)
  }
}

## Mean (+/- 1 SE) of the estimated dimension against the true d.
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
  print(p)
  invisible(p)
}

## Figures of the paper:
# plot_link_comparison(res_main)
# plot_design_comparison(res_main)
# plot_tradeoff(res_main)
# plot_dimension_substudy(res_dim)
# for (m in c("abscos", "TPR", "FDR")) plot_p_scaling(res_scaling, metric = m, n_ref = 100)
