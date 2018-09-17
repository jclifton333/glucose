"""
See https://docs.pymc.io/notebooks/updating_priors.html on updating priors in pymc3.
"""
import numpy as np
import pymc3 as pm
from pymc3 import sample
from pymc3.distributions import Interpolated
from scipy import stats


def update_normal_linear_posterior(prior_parameters, x_new, y_new):
  pass


def from_posterior(param, samples):
  smin, smax = np.min(samples), np.max(samples)
  width = smax - smin
  x = np.linspace(smin, smax, 100)
  y = stats.gaussian_kde(samples)(x)

  # what was never sampled should have a small probability but not 0,
  # so we'll extend the domain and use linear approximation of density on it
  x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
  y = np.concatenate([[0], y, [0]])
  return Interpolated(param, x, y)


def update_nonparametric_posterior(trace):
  model = Model()

  with model:
    # posterior params = from_posterior('param name', trace['param name']
    pass