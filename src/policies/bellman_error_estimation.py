"""
Functions for estimating bellman error using combinations of model-based and model-free temporal difference errors.

Refer to model based and model free BE estimators as delta_mb, delta_mf, respectively.

To get (estimated) MSE-optimal convex combination, we need to estimate
  - sampling variance of delta_mb and delta_mf
  - sampling correlation of delta_mb and delta_mf
  - signal-to-noise ratios of resp sampling dbns
"""
import numpy as np
from policies import expected_q_max


def delta_mf_variance_pooled(TD, number_of_bootstrap_replicates=10):
  """
  Get a bootstrap estimator of the variance of delta_mf, pooling equally across all temporal differences rather than 
  treating each (S, A) pair differently.

  :param TD: array of temporal differences
  :param number_of_bootstrap_replicates:
  :return:
  """
  n = len(TD)
  bootstrapped_TDs = np.zeros((0, n))
  for b in range(number_of_bootstrap_replicates):
    multiplier = np.random.exponential(size=n)
    TD_b = np.multiply(TD, multiplier)
    bootstrapped_TDs = np.vstack((bootstrapped_TDs, TD_b))
  elementwise_variances = np.variance(bootstrapped_TDs, axis=0)
  return np.mean(elementwise_variances)


def delta_mb_variance_pooled(q_fn, env, gamma, X, Sp1, transition_model_fitter, number_of_bootstrap_replicates=10):
  """
  Estimate sampling variance of delta_mb by fitting model to bootstrapped samples of the data (and pooling).

  :param X:
  :param Sp1:
  :param transition_model_fitter:
  :return:
  """
  n = X.shape[0]
  transition_model = transition_model_fitter()
  bootstrapped_TDs = np.zeros((0, n))
  for b in range(number_of_bootstrap_replicates):
    multiplier = np.random.exponential(size=n)
    X_b = np.multiply(multiplier, X_b)
    transition_model_fitter.fit(X_b, Sp1)
    R_b = transition_model_fitter.expected_reward(X)
    expected_q_max_array, _ = expected_q_max(q_fn, X, env, transition_model)
    TD_b = R_b + gamma * expected_q_max_array - q_fn(X)
    bootstrapped_TDs = np.vstack((bootstrapped_TDs, TD_b))
  elementwise_variances = np.variance(bootstrapped_TDs, axis=0)
  return np.mean(elementwise_variances)


def correlation_pooled():
  pass



