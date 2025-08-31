library(marginaleffects)
library(reticulate)

mej <- import("marginaleffectsJAX")

eval_with_python_arrays <- function(FUN, ...) {
  dots <- list(...)
  dots <- lapply(dots, mej$array)
  J <- do.call(FUN, dots)
  J <- mej$array(J)
  return(J)
}

dispatch_function <- function(mfx) {
  if (mfx@calling_function == "predictions") {
    return("predictions")
  } else if (mfx@calling_function == "comparisons") {
    return("comparisons")
  } else {
    stop("Only predictions and comparisons are supported.")
  }
}

dispatch_model <- function(mfx) {
  model <- mfx@model
  if (class(model)[1] == "lm") {
    return("linear")
  } else if (class(model)[1] == "glm") {
    if (model$family$family == "binomial" && model$family$link == "logit") {
      return("logit")
    } else if (model$family$family == "binomial" && model$family$link == "probit") {
      return("probit")
    } else if (model$family$family == "poisson" && model$family$link == "log") {
      return("poisson")
    }
  }
  stop("Only linear, logit, probit, and poisson models are supported.")
}

dispatch_by <- function(mfx) {
  if (isTRUE(mfx@by)) {
    return("_byT")
  } else if (isFALSE(mfx@by)) {
    return("")
  } else {
    stop("Only by = TRUE and by = FALSE are supported.")
  }
  return(estimand)
}

dispatch_estimand <- function(mfx) {
  if (mfx@calling_function == "predictions") {
    return("jacobian")
  } else if (mfx@calling_function == "comparisons") {
    if (mfx@comparison == "difference") {
      return("difference_jacobian")
    } else if (mfx@comparison == "ratio") {
      return("ratio_jacobian")
    } else {
      stop("Only difference and ratio comparisons are supported.")
    }
  }
}

jax_jacobian <- function(coefs, mfx, hi = NULL, lo = NULL, ...) {
  message("\nJAX is fast!")
  f <- dispatch_function(mfx)
  m <- dispatch_model(mfx)
  b <- dispatch_by(mfx)
  e <- dispatch_estimand(mfx)
  args <- list(
    FUN = mej[[m]][[f]][[paste0(e, b)]],
    beta = coefs,
    X = attr(mfx@newdata, "marginaleffects_model_matrix"),
    X_hi = attr(hi, "marginaleffects_model_matrix"),
    X_lo = attr(lo, "marginaleffects_model_matrix")
  )
  args <- Filter(function(x) !is.null(x), args)
  J <- do.call(eval_with_python_arrays, args)
  if (length(dim(J)) == 1) {
    J <- matrix(as.vector(J), nrow = 1)
  }
  return(J)
}

options(marginaleffects_jacobian_function = jax_jacobian)

mod <- glm(am ~ hp + wt, data = mtcars, family = poisson)
mod <- glm(am ~ hp + wt, data = mtcars, family = binomial)
# mod <- glm(am ~ hp + wt, data = mtcars, family = binomial("probit"))

predictions(mod, type = "response", newdata = head(mtcars))

avg_predictions(mod, type = "response")

comparisons(mod, type = "response", newdata = head(mtcars))






fun_jax = function() {
  options(marginaleffects_jacobian_function = jax_jacobian)
  avg_comparisons(mod, type = "response")
}

fun_mfx = function() {
  options(marginaleffects_jacobian_function = NULL)
  avg_comparisons(mod, type = "response")
}

microbenchmark::microbenchmark(
  jax = fun_jax(),
  mfx = fun_mfx(),
  times = 10
)

# options(marginaleffects_jacobian_function = NULL)
# comparisons(mod, type = "response", newdata = head(mtcars))
