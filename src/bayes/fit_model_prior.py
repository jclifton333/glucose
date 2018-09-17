"""
Given (1) a conditional predictive density, (2) a conditional predictive density under the model M_1, and (3)
a conditional predictive density under the model M_2, fit prior \pi_1 such that

Prior(Y) = \pi_1 * Prior(Y | M_1) + (1 - \pi_1) * Prior(Y | M_2)
"""
import numpy as np


def fit_parametric_model_to_glucose_data(X, y):
  # Bayesian linear regression
  pass


def fit_nonparametric_model_to_glucose_data(X, y):
  # Conditional density regression w/ Dirichlet process
  pass


def fit_prior_predictive_density_to_glucose_data(X, y):
  pass


def nonparametric_prior_predictive_density(nonparametric_prior_parameters):
  pass


def parametric_prior_predictive_density(parameteric_prior_parameters):
  pass


def fit_model_prior(predictive_density, model_1_predictive_density, model_2_predictive_density,
                    reference_inputs):
  """

  :param predictive_density:
  :param model_1_predictive_density:
  :param model_2_predictive_density:
  :param reference_inputs:
  :return:
  """
  P = []
  P_1 = []
  P_2 = []

  for x in reference_inputs:
    # Evaluate densities at reference point
    p = predictive_density(x)
    p_1 = model_1_predictive_density(x)
    p_2 = model_2_predictive_density(x)

    P.append(p)
    P_1.append(p_1)
    P_2.append(p_2)

  # Get optimal convex weight (check this)
  p_bar = np.mean(P)
  pi_1 = np.sum((P_2 - P)**2) / np.sum((P - p_bar)**2)

  return pi_1


def fit_prior():
  # Generate random trajectories from glucose data and use these to elicit prior.
  SEED = 3
  np.random.seed(SEED)

  X, y = generate_glucose_data()

  # Get predictive priors
  parametric_prior = fit_parametric_model_to_glucose_data(X, y)
  nonparametric_prior = fit_nonparametric_model_to_glucose_data(X, y)
  predictive_density = fit_prior_predictive_density_to_glucose_data(X, y)
  model_1_predictive_density = parametric_prior_predictive_density(parametric_prior)
  model_2_predictive_density = nonparametric_prior_predictive_density(nonparametric_prior)

  # Get prior model uncertainty
  pi_1 = fit_model_prior(predictive_density, model_1_predictive_density, model_2_predictive_density, X)

  return

