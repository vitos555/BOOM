#   sdy <- sd(y, na.rm = TRUE)
#   ss <- list()
#   sd.prior <- SdPrior(sigma.guess = model.args$prior.level.sd * sdy,
#                       upper.limit = sdy,
#                       sample.size = kLocalLevelPriorSampleSize)
#   ss <- AddLocalLevel(ss, y, sigma.prior = sd.prior)

#   # Add seasonal component?
#   if (model.args$nseasons > 1) {
#     ss <- AddSeasonal(ss, y,
#                       nseasons = model.args$nseasons,
#                       season.duration = model.args$season.duration)
#   }

#   # No regression?
#   if (ncol(data) == 1) {
#     bsts.model <- bsts(y, state.specification = ss, niter = model.args$niter,
#                        seed = 1, ping = 0,
#                        model.options =
#                            BstsOptions(save.prediction.errors = TRUE),
#                        max.flips = model.args$max.flips)
#   } else {
#     formula <- paste0(names(data)[1], " ~ .")

#     # Static regression?
#     if (!model.args$dynamic.regression) {
#       bsts.model <- bsts(formula, data = data, state.specification = ss,
#                          expected.model.size =
#                              kStaticRegressionExpectedModelSize,
#                          expected.r2 = kStaticRegressionExpectedR2,
#                          prior.df = kStaticRegressionPriorDf,
#                          niter = model.args$niter, seed = 1, ping = 0,
#                          model.options =
#                              BstsOptions(save.prediction.errors = TRUE),
#                          max.flips = model.args$max.flips)
#       time(bsts.model$original.series) <- time(data)

#     # Dynamic regression?
#     } else {
#       # Since we have predictor variables in the model, we need to explicitly
#       # make their coefficients time-varying using AddDynamicRegression(). In
#       # bsts(), we are therefore not giving a formula but just the response
#       # variable. We are then using SdPrior to only specify the prior on the
#       # residual standard deviation.
#       # prior.mean: precision of random walk of coefficients
#       sigma.mean.prior <- GammaPrior(prior.mean = 1, a = 4)
#       ss <- AddDynamicRegression(ss, formula, data = data,
#                                  sigma.mean.prior = sigma.mean.prior)
#       sd.prior <- SdPrior(sigma.guess = model.args$prior.level.sd * sdy,
#                           upper.limit = 0.1 * sdy,
#                           sample.size = kDynamicRegressionPriorSampleSize)
#       bsts.model <- bsts(y, state.specification = ss, niter = model.args$niter,
#                          expected.model.size = 3, ping = 0, seed = 1,
#                          prior = sd.prior, max.flips = model.args$max.flips)
#     }
#   }
import pybsts
import numpy as np

class CausalImpact:
  bsts = None

  def __init__(self, x, y, pre_interval, post_interval, seasons=[], dynamic_regression=False, niter=1000, standardize=False):
    specification = {"dynamic_regression": dynamic_regression, "seasons": seasons, "local_trend": {"local_level": True}}
    options = {"niter": niter, "ping": niter/10, "burn": niter/10}
    self.bsts = pybsts.PyBsts("gaussian", x.shape[1], specification, options)

  def standardize(y, fit_range=[]):
    if not fit_range:
      fit_range = range(0, len(y))
    mu = np.mean(y[fit_range, :], axis=0)
    sd = np.std(y[fit_range, :], ddof=1, axis=0)
    y[fit_range, :] = y[fit_range, :] - mu
    y[fit_range, sd > 0] = y[fit_range, sd > 0] / sd[sd > 0]
    def unstandardize(y_):
      y_[fit_range, sd > 0] = y_[fit_range, sd > 0] * sd[sd > 0]
      y_[fit_range, :] = y_[fit_range, :] + mu
      return y_
    return (y, unstandardize)

