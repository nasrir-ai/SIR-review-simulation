## =============================================================================
## real_data_applications.R
##
## Real-data applications of the review paper
##   "Sliced Inverse Regression Revisited: A Comprehensive Review with
##    Simulation and Real Data Applications" (Asrir).
##
## The five estimators of simulation_framework.R are applied to
##   1. Big Mac (alr4::BigMac2003)   n = 69, p = 9      (n > p)
##   2. Riboflavin (hdi::riboflavin)  n = 71, p = 4088   (p >> n)
##
## Three criteria (no true beta is available):
##   - active:       number of non-zero coordinates of beta_hat (|b_j| > 1e-6);
##   - cor_insample: |cor(X beta_hat, y)|;
##   - cv_R2:        cross-validated R^2 of a loess fit of y on the estimated
##                   index, beta_hat being re-estimated on the training folds.
## Cross-validation is paired (the same folds for every method) and repeated
## n_cv_repeats times; cv_R2 is the mean over the repeats (cv_R2_sd its s.d.).
## Each fit is limited to a fixed computing time.
## =============================================================================

## ---- 0. Setup ----------------------------------------------------------------
source("simulation_framework.R")   # packages and the five run_*() wrappers
set.seed(1)

methods <- c("SIR", "SAVE", "LassoSIR", "SIR-NCT", "SIR-WLS")

## ---- 1. Evaluation helpers ----------------------------------------------------

## In-sample metrics for one already-fitted beta_hat.
evaluate_betahat <- function(X, betahat, y, active_tol = 1e-6) {
  betahat <- as.vector(betahat)
  active <- sum(abs(betahat) > active_tol)
  index  <- as.vector(X %*% betahat)
  cor_insample <- if (stats::sd(index) > 0) abs(stats::cor(index, y)) else 0
  list(active = active, cor_insample = cor_insample)
}

## Cross-validated R^2 of one method for a given fold assignment. Falls back
## to the training mean when a fit fails or the estimated index is constant.
cv_r2_single <- function(X, y, fit_fun, folds, timeout_sec = 30, verbose = TRUE) {
  K <- max(folds)
  pred <- numeric(length(y))

  for (k in seq_len(K)) {
    if (verbose) message("      fold ", k, "/", K, " ...", appendLF = FALSE)
    t0 <- Sys.time()
    tr <- folds != k; te <- folds == k
    fallback <- mean(y[tr])

    b <- tryCatch({
      setTimeLimit(elapsed = timeout_sec, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      as.vector(fit_fun(X[tr, , drop = FALSE], y[tr]))
    }, error = function(e) { if (verbose) message(" failed/timed out"); NULL })

    if (is.null(b) || all(b == 0)) { pred[te] <- fallback
      if (verbose) message(sprintf(" fallback (%.1fs)", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      next }

    idx_tr <- as.vector(X[tr, , drop = FALSE] %*% b)
    idx_te <- as.vector(X[te, , drop = FALSE] %*% b)
    if (stats::sd(idx_tr) == 0) { pred[te] <- fallback
      if (verbose) message(" degenerate index, fallback")
      next }

    fit <- tryCatch(
      stats::loess(y ~ idx, data = data.frame(y = y[tr], idx = idx_tr),
                   span = 1, degree = 1, control = stats::loess.control(surface = "direct")),
      error = function(e) NULL)
    p_te <- if (is.null(fit)) rep(NA_real_, length(idx_te)) else
      tryCatch(predict(fit, newdata = data.frame(idx = idx_te)), error = function(e) rep(NA_real_, length(idx_te)))
    p_te[is.na(p_te)] <- fallback
    pred[te] <- p_te
    if (verbose) message(sprintf(" ok (%.1fs)", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }

  1 - sum((y - pred)^2) / sum((y - mean(y))^2)
}

## Applies the five methods to one data set; one row per method.
run_real_data <- function(X, y, H = 5, timeout_sec = 60, K_cv = 5,
                           cv_timeout_sec = 30, n_cv_repeats = 5, label = "") {
  X <- scale(as.matrix(X)); y <- as.vector(y)
  p <- ncol(X)
  n <- nrow(X)

  fit_funs <- list(
    SIR       = function(X, y) run_sir_classic(X, y),
    SAVE      = function(X, y) run_save_classic(X, y),
    LassoSIR  = function(X, y) tryCatch(run_lassosir(X, y, H = H), error = function(e) rep(0, ncol(X))),
    `SIR-NCT` = function(X, y) run_nctsir(X, y, cv = TRUE),
    `SIR-WLS` = function(X, y) run_wlssir(X, y)
  )

  safe_fit <- function(fit_fun) {
    tryCatch({
      setTimeLimit(elapsed = timeout_sec, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      as.vector(fit_fun(X, y))
    }, error = function(e) { message("    failed/timed out: ", conditionMessage(e)); rep(0, p) })
  }

  out <- data.frame(dataset = label, method = methods, n = n, p = p,
                     active = NA_integer_, cor_insample = NA_real_,
                     cv_R2 = NA_real_, cv_R2_sd = NA_real_, Time = NA_real_,
                     stringsAsFactors = FALSE)

  ## ---- Step 1: full-data fit of each method
  ## (a method whose fit fails is recorded with beta_hat = 0)
  main_b <- list(); main_dt <- list(); main_ev <- list()
  for (m in methods) {
    message("-> ", label, " / ", m, " (main fit) ...", appendLF = FALSE)
    t0 <- Sys.time()
    b <- tryCatch(safe_fit(fit_funs[[m]]), error = function(e) {
      message("\n    fit failed: ",
              conditionMessage(e))
      rep(0, p)
    })
    dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    message(sprintf(" done (%.2fs)", dt))
    main_b[[m]] <- b; main_dt[[m]] <- dt; main_ev[[m]] <- evaluate_betahat(X, b, y)
  }

  ## methods whose full-data fit failed are not cross-validated
  active_methods  <- methods[sapply(main_b, function(b) !all(b == 0))]
  skipped_methods <- setdiff(methods, active_methods)
  if (length(skipped_methods) > 0)
    message("      skipping CV for: ", paste(skipped_methods, collapse = ", "),
            " (main fit failed/timed out)")

  ## ---- Step 2: repeated, paired K-fold cross-validation
  cv_mat <- matrix(NA_real_, nrow = n_cv_repeats, ncol = length(active_methods),
                    dimnames = list(NULL, active_methods))

  for (r in seq_len(n_cv_repeats)) {
    folds <- sample(rep(seq_len(K_cv), length.out = n))
    message("    -> CV repeat ", r, "/", n_cv_repeats,
            " (", K_cv, " folds, shared across all ", length(active_methods),
            " active methods this repeat)")
    for (m in active_methods) {
      cv_mat[r, m] <- tryCatch(
        cv_r2_single(X, y, fit_funs[[m]], folds, timeout_sec = cv_timeout_sec, verbose = TRUE),
        error = function(e) NA_real_)
    }
  }

  cv_mean <- colMeans(cv_mat, na.rm = TRUE)
  cv_sd   <- apply(cv_mat, 2, stats::sd, na.rm = TRUE)

  for (m in methods) {
    r2    <- if (m %in% active_methods) unname(cv_mean[m]) else NA_real_
    r2_sd <- if (m %in% active_methods) unname(cv_sd[m])   else NA_real_
    out[out$method == m, c("active", "cor_insample", "cv_R2", "cv_R2_sd", "Time")] <-
      list(main_ev[[m]]$active, main_ev[[m]]$cor_insample, r2, r2_sd, main_dt[[m]])
  }
  out
}

## ---- 2. Big Mac (n = 69, p = 9, low-dimensional) -------------------------------
## alr4::BigMac2003: minutes of labor to buy a Big Mac (response) against
## nine cost-of-living/labor indicators (bread and rice cost, food-price
## index, bus fare, apartment rent, teacher gross/net income, tax rate,
## teacher hours) for 69 cities.
if (!requireNamespace("alr4", quietly = TRUE)) install.packages("alr4")
data(BigMac2003, package = "alr4")
bm   <- na.omit(BigMac2003)
y_bm <- bm$BigMac
X_bm <- as.matrix(bm[, setdiff(names(bm), "BigMac")])

res_bigmac <- run_real_data(X_bm, y_bm, H = 5, n_cv_repeats = 5, label = "BigMac")
print(res_bigmac)

## ---- 3. Riboflavin (n = 71, p = 4088, p >> n) ----------------------------------
## hdi::riboflavin: log riboflavin production rate (response) against log
## expression of 4088 genes, 71 bacterial cultures.
if (!requireNamespace("hdi", quietly = TRUE)) install.packages("hdi")
data(riboflavin, package = "hdi")
y_rib <- riboflavin$y
X_rib <- as.matrix(riboflavin$x)

## 3-fold cross-validation and shorter time limits (p = 4088).
res_ribo <- run_real_data(X_rib, y_rib, H = 5, timeout_sec = 120,
                          K_cv = 3, cv_timeout_sec = 45, n_cv_repeats = 5,
                          label = "Riboflavin")
print(res_ribo)

## ---- 4. Combined table ---------------------------------------------------------

res_realdata <- rbind(res_bigmac, res_ribo)
write.csv(res_realdata, "sir_review_realdata_results.csv", row.names = FALSE)
print(res_realdata)
