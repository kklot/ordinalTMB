
<!-- README.md is generated from README.Rmd. Please edit that file -->

# ordinalTMB - univariate and multivariate ordinal regresison

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

This package is written for the “Gender Across Spectrum” paper (doi to
be updated), expecting to be reused further for Likert scales outcomes
studies.

## Why?

Run an ordinal model with random effects, but:

- `ordinal::clmm` sometimes gave all the parameter estimates the same
  SE, not sure why; nevertheless, the ultimate aim is to do Bayesian
  inference
- `rstanarm::stan_polr` does not provide random effect specification (?)
- `brms::brm()` with `cummulative` family seems like the one, but failed
  to compile on MacOS, or an HPC server (scientific linux), and on both
  systems with a separated `miniconda` environment.
  - on both systems, other packages using Eigen, Armadillo, Boost,
    INLA,…work fine.
  - the data is large, so `brm` would be slow anyway
- `INLA` [does not have ordinal model since
  2015?](https://groups.google.com/g/r-inla-discussion-group/c/7Tl6DanHmtM/m/agDFK0w23VAJ)

Moreover, to do multivariates ordinal regression

- similar to `mvord`, `PLordprob`, which do not have random effect
  specification

## TODO

By priorities

- [x] Fit a simple ordinal with logistic link and compare with `clm`
  estimates
- [x] Fit an ordinal logistic with random intercepts and compared with
  `clmm`
- [x] add probit link and compare with `clmm`, `clm`
- [ ] add multivariates ordinal
- [ ] add other links (c-log-log,…)
- [ ] add glm-ism formula interface?

## Temporarily workflow - univariate ordinal

1.  Clone the package
2.  Prepare data and meta parameters for TMB like usual, example:

``` r
data <- full_data |> na.omit()
kdata = list()
kdata$y <- data$outcome # coded as 1, 2,... as its order
fm <- y ~ -1 + q + time + age # no intercept and with covariates
# Fix effect
mf <- model.frame(fm, data)
kdata$X <- model.matrix(fm, mf)
(J <- max(kdata$y)) # number of categories
(K <- ncol(kdata$X)) # number of covariates
# random effect
kdata$R <- make_iid_matrix(data$id)
(S <- length(unique(data$id)))
# TMB parameters
kpar <- list(
  pi_norm = rnorm(J - 1),
  beta = rnorm(K),
  iid = rnorm(S), 
  sd_log = log(1)
)
```

2.  Edit the TMB model files in `ordinaTMB/src/ordinalTMB.cpp` according
    to your need

``` r
ktools::tmb_unload('ordinalTMB'); # run this every time you edit the model file
devtools::load_all("~/ordinalTMB/") # load the package
model <- ktools::MakeADFunSafe( # avoid crashing R's session
    data = kdata,
    parameters = kpar,
    random = c('iid'),
    DLL = "ordinalTMB"
)
fit <- nlminb(model$par, model$fn, model$gr)
model$report() |> glimpse()
```
