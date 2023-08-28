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