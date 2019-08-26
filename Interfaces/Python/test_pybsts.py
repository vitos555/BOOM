import pybsts
import numpy as np

state_space_model = pybsts.StateSpaceModel()
observation_model = state_space_model.observation_model()
chisq_model = pybsts.ChisqModel(1, 0.5)
sigma_sampler = pybsts.ZeroMeanGaussianConjSampler(observation_model, chisq_model)
observation_model.set_method(sigma_sampler)
state_space_posterior_sampler = pybsts.StateSpacePosteriorSampler(state_space_model)
state_space_model.set_method(state_space_posterior_sampler)
ys = [1.0, 2.0, 3.0, 4.0, 4.5]
for y in ys:
	state_space_model.add_data(pybsts.MultiplexedDoubleData(y))
static_intercept = pybsts.StaticInterceptStateModel()
static_intercept.set_initial_state_mean(np.mean(ys))
static_intercept.set_initial_state_variance(np.square(np.std(ys, ddof=1)))
state_space_model.add_state(static_intercept)
state_space_model.sample_posterior()
for iteraion in range(0, 1000):
	state_space_model.sample_posterior()
forecast=state_space_model.forecast(1)
print(forecast.size())
print(forecast.nrow())
for i in forecast:
	print(i)