
<!-- README.md is generated from README.Rmd. Please edit that file -->

# ordinalTMB

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

## Why?

I wanted to run an ordinal model with random effects, but for my data:

- `ordinal::clmm` gives all the parameter estimates the same SE, not
  sure why
- `rstanarm::stan_polr` does not provide random effect specification (is
  this true?)
- `brms::brm()` with `cummulative` family seems like the one, but I
  cannot get it to compile on my MacOS, or on my HPC server with
  scientific linux, and on both systems with a separated `miniconda`
  environment.
  - all both systems, other packages using Eigen, Armadillo, Boost,
    INLA,…work fine.

## TODO

- [x] Fit a simple ordinal with logistic link and compare with `clm`’s
  estimates
- [x] Fit an ordinal logistic with random intercepts and compared with
  `clmm`
- [ ] add other links (logit, c-log-log,…)?
- [ ] add glm-ism formula interface?

## Workflow

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
