import pybsts
import numpy as np

def identity_fn(y, transform_range=[]):
    if len(y.shape) == 1:
        return (y[transform_range], lambda y, r: y[r], lambda y, r: y[r])
    else:
        return (y[transform_range, :], lambda y, r: y[r, :], lambda y, r: y[r, :])

def standardize_fn(y_, transform_range=[]):
    y = np.array(y_)
    if len(y.shape) == 1:
        if not transform_range:
            transform_range = range(0, len(y))
        mu = np.mean(y[transform_range])
        sd = np.std(y[transform_range], ddof=1)
        y[transform_range] = y[transform_range] - mu
        y[transform_range] = y[transform_range] / sd if sd > 0 else y[transform_range]
        def restandardize_fn(y_, transform_range):
            y = np.array(y_)
            y[transform_range] = y[transform_range] - mu
            y[transform_range] = y[transform_range] / sd if sd > 0 else y[transform_range]
            return y
        def unstandardize_fn(y_, transform_range):
            y = np.array(y_)
            y[transform_range] = y[transform_range] * sd if sd > 0 else y[transform_range]
            y[transform_range] = y[transform_range] + mu
            return y
    else:
        if not transform_range:
            transform_range = range(0, y.shape[0])
        mu = np.mean(y[transform_range, :], axis=0)
        sd = np.std(y[transform_range, :], ddof=1, axis=0)
        y[transform_range, :] = y[transform_range, :] - mu
        y[transform_range, sd > 0] = y[transform_range, sd > 0] / sd[sd > 0]
        def restandardize_fn(y_, transform_range):
            y = np.array(y_)
            y[transform_range, sd > 0] = y[transform_range, sd > 0] * sd[sd > 0]
            y[transform_range, :] = y[transform_range, :] + mu
            return y
        def unstandardize_fn(y_, transform_range):
            y = np.array(y_)
            y[transform_range, sd > 0] = y[transform_range, sd > 0] * sd[sd > 0]
            y[transform_range, :] = y[transform_range, :] + mu
            return y
    return (y, restandardize_fn, unstandardize_fn)

class CausalImpact:
    bsts = None
    transformation_fn = None
    niter = 1000
    alpha = 0.05
    burn = 100
    x = None
    y = None
    pre_period = None
    post_period = None
    results = None

    def __init__(self, x, y, pre_period, post_period,
                 predictor_names=None, seasons=[], dynamic_regression=False,
                 niter=1000, burn=100, standardize=False, alpha=0.05, seed=None):
        if not predictor_names:
            predictor_names = ["intercept"] + ["x" + str(i) for i in range(0, x.shape[0])]
        x = np.insert(x, 0, np.repeat(1.0, x.shape[1]), axis=0).T
        specification = {"dynamic_regression": dynamic_regression,
                         "seasons": seasons,
                         "predictor_names": predictor_names,
                         "initial_value": y[0],
                         "mean_value": np.mean(y[pre_period]),
                         "sigma_prior": np.std(y[pre_period], ddof=1),
                         "predictors_prior": {"predictors_squared_normalized": np.dot(x.T, x)/x.shape[0]},
                         "local_trend": {"local_level": True}}
        options = {"niter": niter, "ping": niter/10, "burn": burn}
        if seed:
            options["seed"] = seed
        self.bsts = pybsts.PyBsts("gaussian", specification, options)
        self.transformation_fn = identity_fn
        if standardize:
            self.transformation_fn = standardize_fn
        self.niter = niter
        self.burn = burn
        self.alpha = alpha
        self.x = x
        self.y = y
        self.pre_period = pre_period
        self.post_period = post_period

    def analyze(self):
        if self.results is not None:
            return self.results
        training_y, retransform_y, untransform_y = \
            self.transformation_fn(self.y, self.pre_period)
        training_x, retransform_x, untransform_x = \
            self.transformation_fn(self.x, self.pre_period)
        prediction_y = retransform_y(self.y, self.post_period)
        prediction_x = retransform_x(self.x, self.post_period)
        input_x = np.concatenate((training_x, prediction_x), axis=0)
        input_y = np.concatenate((training_y, np.repeat(np.nan, self.post_period[-1]-self.post_period[0]+1)))
        observed_y = [True]*(self.pre_period[-1]-self.pre_period[0]+1) + \
            [False]*(self.post_period[-1]-self.post_period[0]+1)
        self.bsts.fit(input_x, input_y, observed=observed_y)
        state_contributions = np.array(self.bsts.results("state.contributions"))
        # print(state_contributions)
        state_contributions.shape = (input_x.shape[0], -1, self.niter)
        state_contributions = np.transpose(state_contributions, axes=[2, 0, 1])
        # print(state_contributions)
        # print(np.sum(state_contributions, axis=2)[1, :])
        pred_mean = np.sum(state_contributions, axis=2)[self.burn:, :]
        print(pred_mean[98:100:, :])
        sigma_obs = self.bsts.results("sigma.obs")[self.burn:]
        sigma_obs = np.reshape(np.repeat(sigma_obs, pred_mean.shape[1]),
                               pred_mean.shape)
        noise = np.random.normal(0, sigma_obs)
        pred = pred_mean + noise
        pred_mean = np.mean(pred_mean, axis=0)
        pred_lower = np.quantile(pred, self.alpha/2, axis=0)
        pred_upper = np.quantile(pred, 1-self.alpha/2, axis=0)
        actuals = np.concatenate((training_y, prediction_y))
        self.results = (untransform_y(actuals, np.array(list(self.pre_period) + list(self.post_period))),
                untransform_y(pred_mean, np.array(list(self.pre_period) + list(self.post_period))),
                untransform_y(pred_lower, np.array(list(self.pre_period) + list(self.post_period))),
                untransform_y(pred_upper, np.array(list(self.pre_period) + list(self.post_period))),
                untransform_y(pred, np.array(list(self.pre_period) + list(self.post_period))))
        return self.results

    def summary_dict(self):
        if self.results is None:
            self.analyze()
        results = self.results
        summary = {"average": {}, "cumulative": {}}
        post_actual = results[0][self.post_period]
        summary["average"]["actual"] = np.mean(post_actual)
        summary["cumulative"]["actual"] = np.sum(post_actual)

        post_predicted = results[1][self.post_period]
        summary["average"]["predicted"] = np.mean(post_predicted)
        summary["cumulative"]["predicted"] = np.sum(post_predicted)

        post_pred = results[4][:, self.post_period]
        post_actual_rep = np.repeat(np.reshape(post_actual, (1, -1)), post_pred.shape[0], axis=0)
        post_actual_rep.shape = post_pred.shape

        summary["average"]["predicted_lower"] = np.quantile(np.mean(post_pred, axis=0), self.alpha/2)
        summary["cumulative"]["predicted_lower"] = np.quantile(np.sum(post_pred, axis=0), self.alpha/2)

        summary["average"]["predicted_upper"] = np.quantile(np.mean(post_pred, axis=0), 1.0 - self.alpha/2)
        summary["cumulative"]["predicted_upper"] = np.quantile(np.sum(post_pred, axis=0), 1.0 - self.alpha/2)

        summary["average"]["predicted_std"] = np.std(np.mean(post_pred, axis=0), ddof=1)
        summary["cumulative"]["predicted_std"] = np.std(np.sum(post_pred, axis=0), ddof=1)

        summary["average"]["abs_effect"] = np.mean(post_actual) - np.mean(post_predicted)
        summary["cumulative"]["abs_effect"] = np.sum(post_actual) - np.sum(post_predicted)

        summary["average"]["abs_effect_lower"] = \
            np.quantile(np.mean(post_actual_rep - post_pred, axis=0), self.alpha/2)
        summary["cumulative"]["abs_effect_lower"] = \
            np.quantile(np.sum(post_actual_rep - post_pred, axis=0), self.alpha/2)

        summary["average"]["abs_effect_upper"] = \
            np.quantile(np.mean(post_actual_rep - post_pred, axis=0), 1.0 - self.alpha/2)
        summary["cumulative"]["abs_effect_upper"] = \
            np.quantile(np.sum(post_actual_rep - post_pred, axis=0), 1.0 - self.alpha/2)

        summary["average"]["abs_effect_std"] = np.std(np.mean(post_actual_rep - post_pred, axis=0))
        summary["cumulative"]["abs_effect_std"] = np.std(np.sum(post_actual_rep - post_pred, axis=0))

        summary["average"]["rel_effect"] = \
            summary["average"]["abs_effect"] / summary["average"]["predicted"]
        summary["cumulative"]["rel_effect"] = \
            summary["cumulative"]["abs_effect"] / summary["cumulative"]["predicted"]

        summary["average"]["rel_effect_lower"] = \
            summary["average"]["abs_effect_lower"] / summary["average"]["predicted"]
        summary["cumulative"]["rel_effect_lower"] = \
            summary["cumulative"]["abs_effect_lower"] / summary["cumulative"]["predicted"]

        summary["average"]["rel_effect_upper"] = \
            summary["average"]["abs_effect_upper"] / summary["average"]["predicted"]
        summary["cumulative"]["rel_effect_upper"] = \
            summary["cumulative"]["abs_effect_upper"] / summary["cumulative"]["predicted"]

        summary["average"]["rel_effect_std"] = \
            summary["average"]["abs_effect_std"] / summary["average"]["predicted"]
        summary["cumulative"]["rel_effect_std"] = \
            summary["cumulative"]["abs_effect_std"] / summary["cumulative"]["predicted"]

        post_pred_row_sums = np.sum(post_pred, axis=0)
        post_actual_sum = np.sum(post_actual)
        summary["cumulative"]["pvalue"] = min(np.sum(post_pred_row_sums >= post_actual_sum) + 1,
                                              np.sum(post_pred_row_sums <= post_actual_sum) + 1) / \
                                          (len(post_pred_row_sums) + 1)
        summary["cumulative"]["significance"] = (1-summary["cumulative"]["pvalue"])*100

        return summary

    def summary(self):
        summary_dict = self.summary_dict()
        summary_dict["CI"] = (1.0-self.alpha)*100
        for col in summary_dict["average"]:
            if col[0:4] == "rel_":
                summary_dict["average"][col] *= 100
        for col in summary_dict["cumulative"]:
            if col[0:4] == "rel_":
                summary_dict["cumulative"][col] *= 100

        out = """
Posterior inference {{CausalImpact}}

                         Average            Cumulative  
Actual                   {average[actual]:7.0f}               {cumulative[actual]:7.0f}         
Prediction (s.d.)        {average[predicted]: 7.1f} ({average[predicted_std]:.1f})             {cumulative[predicted]:.1f} ({cumulative[predicted_std]:.1f})  
{CI:2.0f}% CI                [{average[predicted_lower]:6.1f}, {average[predicted_upper]:6.1f}]         [{cumulative[predicted_lower]:6.1f}, {cumulative[predicted_upper]:6.1f}] 
                                                    
Absolute effect (s.d.)   {average[abs_effect]:7.1f} ({average[abs_effect_std]:.1f})         {cumulative[abs_effect]:7.1f} ({cumulative[abs_effect_std]:.1f})
{CI:2.0f}% CI                [{average[abs_effect_lower]:6.1f}, {average[abs_effect_upper]:6.1f}]         [{cumulative[abs_effect_lower]:6.1f}, {cumulative[abs_effect_upper]:6.1f}] 
                                                    
Relative effect (s.d.)   {average[rel_effect]:7.2f}% ({average[rel_effect_std]:.2f}%)   {cumulative[rel_effect]:7.2f}% ({cumulative[rel_effect_std]:.2f}%)
{CI:2.0f}% CI                   [{average[rel_effect_lower]:5.2f}%, {average[rel_effect_upper]:5.2f}%]    [{cumulative[rel_effect_lower]:5.2f}%, {cumulative[rel_effect_upper]:5.2f}%] 

Posterior tail-area probability p:   {cumulative[pvalue]:.5f}
Posterior prob. of a causal effect:  {cumulative[significance]:.2f}%
"""
        return out.format(**summary_dict)
