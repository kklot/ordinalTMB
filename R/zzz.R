#' ordinalTMB: ordinal regression with TMB
#'
#' @section To model Likert scale outcome.
#'
#' @docType package
#' @name ordinalTMB
#' @useDynLib ordinalTMB, .registration=TRUE
NULL

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("Welcome to ordinalTMB")
}

.onUnload <- function(libpath) {
  library.dynam.unload("ordinalTMB", libpath)
}