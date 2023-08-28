# simplex transform
invlogit = function (x) 1/(1 + exp(-x))

#' transform an unbounded (normally distributed) to simplex
#'
# https://mc-stan.org/docs/reference-manual/simplex-transform.html
#' 
#' @param y a vector of lenth J - 1
#' @return an simplex of length J
#' @examples
#' sum(simplex_transform(rnorm(5)))
#' @export
simplex_transform = function(y) {
	K = length(y) + 1
	z = rep(0, K-1)
	for (k in 1:(K-1))
		z[k] = invlogit(y[k] + log(1/(K - k)))
	x = rep(0, K)
	x[1] = z[1]
	for (k in 2:(K-1))
		x[k] = (1 - sum(x[1:(k-1)]) ) * z[k]
	x[K] = 1 - sum(x[1:(K-1)])
	x
}
#' transform unbounded to unit vector
#' @param y numeric vector
#' @examples
#' x = as_unit_vector(rnorm(5))
#' (sqrt(x %*% x) == 1)
#' @export
as_unit_vector = function(y) y/(sqrt(y %*% y)[1,1])

#' make design matrix for IID random effect
#'
#' @param x subject ID column
#' @export
make_iid_matrix <- function (x) {
    require(Matrix)
    mm <- unique(x)
    id <- match(x, mm)
    ma <- matrix(0, length(x), length(mm))
    ma[cbind(1:length(x), id)] <- 1
    re <- as(ma, "sparseMatrix")
    re
}


#' Sample multivariates with precision matrix
#' 
#' @param n number of samples 
#' @param mean scalar/vector of the mean(s) 
#' @param prec precision matrix
#' @return n samples 
#' @export 
rmvnorm_sparseprec <- function(n, mean = rep(0, nrow(prec)), prec = diag(lenth(mean))) {
  z <- matrix(rnorm(n * length(mean)), ncol = n)
  L_inv <- Matrix::Cholesky(prec)
  v <- mean +
    Matrix::solve(
      as(L_inv, "pMatrix"),
      Matrix::solve(
        Matrix::t(as(L_inv, "Matrix")), z
      )
    )
  as.matrix(Matrix::t(v))
}

# y=z
# y = backsolve(chol(prec), z)
# L_inv = Matrix::Cholesky(prec, perm=FALSE, LDL=FALSE)

#' Sample TMB fit
#'
#' @param fit The TMB fit
#' @param nsample Number of samples
#' @param random_only Random only
#' @param verbose If TRUE prints additional information.
#'
#' @return Sampled fit.
#' @export
sample_tmb <- function(fit, nsample = 1000, random_only = TRUE, verbose = TRUE) {
  to_tape <- TMB:::isNullPointer(fit$obj$env$ADFun$ptr)
  if (to_tape) {
    message("Retaping...")
    obj <- fit$obj
    fit$obj <- with(
      obj$env,
      TMB::MakeADFun(
        data,
        parameters,
        map = map,
        random = random,
        silent = silent,
        DLL = DLL
      )
    )
    fit$obj$env$last.par.best <- obj$env$last.par.best
  } else {
    message("No taping done.")
  }
  par.full <- fit$obj$env$last.par.best
  if (!random_only) {
    if (verbose) print("Calculating joint precision")
    hess <- TMB::sdreport(fit$obj, fit$fit$par, getJointPrecision = TRUE)
    if (verbose) print("Drawing sample")
    smp <- rmvnorm_sparseprec(nsample, par.full, hess)
  } else {
    r_id <- fit$obj$env$random
    par_f <- par.full[-r_id]
    par_r <- par.full[r_id]
    hess_r <- fit$obj$env$spHess(par.full, random = TRUE)
    smp_r <- rmvnorm_sparseprec(nsample, par_r, hess_r)
    smp <- matrix(0, nsample, length(par.full))
    smp[, r_id] <- smp_r
    smp[, -r_id] <- matrix(par_f, nsample, length(par_f), byrow = TRUE)
    colnames(smp) <- names(par.full)
  }
  smp
}