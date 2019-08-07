devtools::load_all(path="Boom")
devtools::load_all(path="BoomSpikeSlab")
devtools::load_all(path="bsts")

x <- data.frame(y <- c(1.0, 2.0, 3.0, 4.0, 4.5, NA, NA), a=c(1.0, 2.0, 0.0, 0.0, 0.0, 3.5, 0.0), b=c(0.0, 0.0, 3.0, 4.0, 4.4, 0.0, 2.5))
ss <- AddLocalLevel(list(), y)
m <- bsts(y ~ a + b, data=x, ss, seed=1, niter=1000)

alpha <- 0.05
state.contributions <- m$state.contributions[-seq_len(100), , , drop = FALSE]
state.samples <- rowSums(aperm(state.contributions, c(1, 3, 2)), dims = 2)

sigma.obs <- m$sigma.obs[-seq_len(100)]
n.samples <- dim(state.samples)[1]

obs.noise.samples <- matrix(rnorm(prod(dim(state.samples)), 0, sigma.obs),
                              nrow = n.samples)

y.samples <- state.samples + obs.noise.samples
point.pred.mean <- colMeans(state.samples)
prob.lower <- alpha / 2      # e.g., 0.025 when alpha = 0.05
prob.upper <- 1 - alpha / 2  # e.g., 0.975 when alpha = 0.05
point.pred.lower <- as.numeric(t(apply(y.samples, 2, quantile, prob.lower)))
point.pred.upper <- as.numeric(t(apply(y.samples, 2, quantile, prob.upper)))