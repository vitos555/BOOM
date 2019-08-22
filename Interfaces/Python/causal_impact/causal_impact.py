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
    mid_period = None
    results = None
    seed = None

    def __init__(self, x, y, pre_period, post_period,
                 predictor_names=None, seasons=[], dynamic_regression=False,
                 niter=1000, burn=100, standardize=False, alpha=0.05, ping=None, seed=None):
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
        options = {"niter": niter, "ping": niter/10 if ping is None else ping, "burn": burn}
        if seed:
            options["seed"] = seed
            self.seed = seed + 1
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
        self.mid_period = range(0, 0)
        if min(self.post_period) - max(self.pre_period) > 0:
            self.mid_period = range(max(self.pre_period) + 1, min(self.post_period))

    def analyze(self):
        if self.results is not None:
            return self.results
        training_y, retransform_y, untransform_y = \
            self.transformation_fn(self.y, self.pre_period)
        training_x, retransform_x, untransform_x = \
            self.transformation_fn(self.x, self.pre_period)
        prediction_y = retransform_y(self.y, self.post_period)
        prediction_x = retransform_x(self.x, self.post_period)
        mid_hold_y = retransform_y(self.y, self.mid_period)
        mid_hold_x = retransform_x(self.x, self.mid_period)
        input_x = np.concatenate((training_x, prediction_x), axis=0)
        input_y = np.concatenate((training_y, np.repeat(np.nan, self.post_period[-1]-self.post_period[0]+1)))
        observed_y = [True]*(self.pre_period[-1]-self.pre_period[0]+1) + \
            [False]*(self.post_period[-1]-self.post_period[0]+1)
        self.bsts.fit(input_x, input_y, observed=observed_y, seed=self.seed)
        adjusted_post_period = np.array(self.post_period) - (min(self.post_period) - max(self.pre_period) - 1)
        state_contributions = np.array(self.bsts.results("state.contributions"))
        state_contributions.shape = (input_x.shape[0], -1, self.niter)
        state_contributions = np.transpose(state_contributions, axes=[2, 0, 1])
        pred_mean = np.sum(state_contributions, axis=2)[self.burn:, :]
        sigma_obs = self.bsts.results("sigma.obs")[self.burn:]
        sigma_obs = np.reshape(np.repeat(sigma_obs, pred_mean.shape[1]),
                               pred_mean.shape)
        np.random.seed(self.seed)
        noise = np.random.normal(0, sigma_obs)
        nan_mid_y = np.repeat(np.nan, mid_hold_y.shape[0])
        pred = pred_mean + noise
        pred_mean = np.mean(pred_mean, axis=0)
        pred_lower = np.quantile(pred, self.alpha/2, axis=0)
        pred_upper = np.quantile(pred, 1-self.alpha/2, axis=0)
        actuals = np.concatenate((training_y, mid_hold_y, prediction_y))
        pred_mean = np.concatenate((pred_mean[self.pre_period], nan_mid_y, pred_mean[adjusted_post_period]))
        pred_lower = np.concatenate((pred_lower[self.pre_period], nan_mid_y, pred_lower[adjusted_post_period]))
        pred_upper = np.concatenate((pred_upper[self.pre_period], nan_mid_y, pred_upper[adjusted_post_period]))
        pred = np.concatenate((pred[:, self.pre_period], np.repeat(np.reshape(nan_mid_y, (1, -1)), pred.shape[0], axis=0), pred[:, adjusted_post_period]), axis=1)
        self.results = (untransform_y(actuals, np.array(list(self.pre_period) + list(self.mid_period) + list(self.post_period))),
                untransform_y(pred_mean, np.array(list(self.pre_period) + list(self.mid_period) + list(self.post_period))),
                untransform_y(pred_lower, np.array(list(self.pre_period) + list(self.mid_period) + list(self.post_period))),
                untransform_y(pred_upper, np.array(list(self.pre_period) + list(self.mid_period) + list(self.post_period))),
                untransform_y(pred.T, np.array(list(self.pre_period) + list(self.mid_period) + list(self.post_period))))
        return self.results

    def summary_dict(self):
        if self.results is None:
            self.analyze()
        results = self.results
        summary_obj = {"average": {}, "cumulative": {}}
        post_actual = results[0][self.post_period]
        summary_obj["average"]["actual"] = np.mean(post_actual)
        summary_obj["cumulative"]["actual"] = np.sum(post_actual)

        post_predicted = results[1][self.post_period]
        summary_obj["average"]["predicted"] = np.mean(post_predicted)
        summary_obj["cumulative"]["predicted"] = np.sum(post_predicted)

        post_pred = results[4][self.post_period, :]
        post_actual_rep = np.repeat(np.reshape(post_actual, (-1, 1)), post_pred.shape[1], axis=1)

        summary_obj["average"]["predicted_lower"] = np.quantile(np.mean(post_pred, axis=0), self.alpha/2)
        summary_obj["cumulative"]["predicted_lower"] = np.quantile(np.sum(post_pred, axis=0), self.alpha/2)

        summary_obj["average"]["predicted_upper"] = np.quantile(np.mean(post_pred, axis=0), 1.0 - self.alpha/2)
        summary_obj["cumulative"]["predicted_upper"] = np.quantile(np.sum(post_pred, axis=0), 1.0 - self.alpha/2)

        summary_obj["average"]["predicted_std"] = np.std(np.mean(post_pred, axis=0), ddof=1)
        summary_obj["cumulative"]["predicted_std"] = np.std(np.sum(post_pred, axis=0), ddof=1)

        summary_obj["average"]["abs_effect"] = np.mean(post_actual) - np.mean(post_predicted)
        summary_obj["cumulative"]["abs_effect"] = np.sum(post_actual) - np.sum(post_predicted)

        summary_obj["average"]["abs_effect_lower"] = \
            np.quantile(np.mean(post_actual_rep - post_pred, axis=0), self.alpha/2)
        summary_obj["cumulative"]["abs_effect_lower"] = \
            np.quantile(np.sum(post_actual_rep - post_pred, axis=0), self.alpha/2)

        summary_obj["average"]["abs_effect_upper"] = \
            np.quantile(np.mean(post_actual_rep - post_pred, axis=0), 1.0 - self.alpha/2)
        summary_obj["cumulative"]["abs_effect_upper"] = \
            np.quantile(np.sum(post_actual_rep - post_pred, axis=0), 1.0 - self.alpha/2)

        summary_obj["average"]["abs_effect_std"] = np.std(np.mean(post_actual_rep - post_pred, axis=0))
        summary_obj["cumulative"]["abs_effect_std"] = np.std(np.sum(post_actual_rep - post_pred, axis=0))

        summary_obj["average"]["rel_effect"] = \
            summary_obj["average"]["abs_effect"] / summary_obj["average"]["predicted"]
        summary_obj["cumulative"]["rel_effect"] = \
            summary_obj["cumulative"]["abs_effect"] / summary_obj["cumulative"]["predicted"]

        summary_obj["average"]["rel_effect_lower"] = \
            summary_obj["average"]["abs_effect_lower"] / summary_obj["average"]["predicted"]
        summary_obj["cumulative"]["rel_effect_lower"] = \
            summary_obj["cumulative"]["abs_effect_lower"] / summary_obj["cumulative"]["predicted"]

        summary_obj["average"]["rel_effect_upper"] = \
            summary_obj["average"]["abs_effect_upper"] / summary_obj["average"]["predicted"]
        summary_obj["cumulative"]["rel_effect_upper"] = \
            summary_obj["cumulative"]["abs_effect_upper"] / summary_obj["cumulative"]["predicted"]

        summary_obj["average"]["rel_effect_std"] = \
            summary_obj["average"]["abs_effect_std"] / summary_obj["average"]["predicted"]
        summary_obj["cumulative"]["rel_effect_std"] = \
            summary_obj["cumulative"]["abs_effect_std"] / summary_obj["cumulative"]["predicted"]

        post_pred_row_sums = np.sum(post_pred, axis=0)
        post_actual_sum = np.sum(post_actual)
        summary_obj["cumulative"]["pvalue"] = (min(np.sum(post_pred_row_sums >= post_actual_sum),
                                              np.sum(post_pred_row_sums <= post_actual_sum) ) + 1) / \
                                          (len(post_pred_row_sums) + 1)
        summary_obj["cumulative"]["significance"] = (1-summary_obj["cumulative"]["pvalue"])*100

        return summary_obj

    def lines(self, aslists=False, precision=None):
        self.analyze()
        results = self.results
        actual = results[0]
        pred = results[4]
        actual_rep =  np.repeat(np.reshape(actual, (-1, 1)), pred.shape[1], axis=1)
        ret = {"point_estimates": {}, "cum_diff": {}, "abs_diff": {}}
        ret["point_estimates"]["actuals"] = results[0]
        ret["point_estimates"]["pred_mean"] = results[1]
        ret["point_estimates"]["pred_lower"] = results[2]
        ret["point_estimates"]["pred_upper"] = results[3]
        ret["abs_diff"]["mean"] = np.mean(actual_rep - pred, axis=1)
        ret["abs_diff"]["lower"] = np.quantile(actual_rep - pred, self.alpha/2, axis=1)
        ret["abs_diff"]["upper"] = np.quantile(actual_rep - pred, 1-self.alpha/2, axis=1)
        ret["cum_diff"]["mean"] = np.cumsum(ret["abs_diff"]["mean"])
        ret["cum_diff"]["lower"] = np.cumsum(ret["abs_diff"]["lower"])
        ret["cum_diff"]["upper"] = np.cumsum(ret["abs_diff"]["upper"])
        if precision is not None:
            for parent in ret:
                for key in ret[parent]:
                    ret[parent][key] = np.around(ret[parent][key], decimals=precision)
        if aslists:
            for parent in ret:
                for key in ret[parent]:
                    ret[parent][key] = list(ret[parent][key])
        return ret

    def summary(self):
        summary_obj = self.summary_dict()
        summary_obj["CI"] = (1.0-self.alpha)*100
        for col in summary_obj["average"]:
            if col[0:4] == "rel_":
                summary_obj["average"][col] *= 100
        for col in summary_obj["cumulative"]:
            if col[0:4] == "rel_":
                summary_obj["cumulative"][col] *= 100

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
        return out.format(**summary_obj)
