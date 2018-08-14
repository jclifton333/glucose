import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(this_dir, '..', 'src')
results_dir = os.path.join(this_dir, '..', 'analysis', 'results')

from src.policies.policies import model_smoothed_fitted_q
from src.Glucose import Glucose
from src.transition_estimation import MultivariateLinear
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime
import yaml
import multiprocessing as mp


def fitted_q_with_model_smoothing(index):

  env = Glucose(horizon=30)
  gamma = 0.9
  number_of_value_iterations = 1
  number_of_replicates = 30
  transition_model_fitter = MultivariateLinear

  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  basename = 'mse_smooth_{}_rep_{}_valiter'.format(number_of_replicates, number_of_value_iterations)
  fname = os.path.join(results_dir, basename, suffix, '.yml')

  scores = []
  results = {'alpha_mb': [], 'mb_variances': [], 'mb_biases': [], 'mf_variances': [], 'score': None}
  env.reset()
  done = False
  episode_score = 0.0
  while not done:
    action, pairwise_kernel_, kernel_sums, bootstrapped_mse_components_, alpha_mb = \
      model_smoothed_fitted_q(env, gamma, RandomForestRegressor, number_of_value_iterations, transition_model_fitter,
                              pairwise_kernels_=None, kernel_sums=None, smoothing_method='mse')
    _, r, done = env.step(action)
    episode_score += r

    results['alpha_mb'].append(alpha_mb)
    results['mb_variances'].append(bootstrapped_mse_components_['mb_variances'])
    results['mf_variances'].append(bootstrapped_mse_components_['mf_variances'])
    results['mb_biases'].append(bootstrapped_mse_components_['mf_biases'])

  results['score'] = episode_score
  results[rep] = results
  scores.append(episode_score)
  print('score: {}'.format(score))
  results['mean score'] = np.mean(scores)
  results['se score'] = np.std(scores) / np.sqrt(number_of_replicates)
  with open(fname, 'w') as outfile:
    yaml.dump(results, outfile)
  return


if __name__ == '__main__':
  n_replicate = 30
  pool = mp.Pool(int(np.min(n_replicate, mp.cpu_count() / 2)))
  pool.map(fitted_q_with_model_smoothing, range(n_replicate))
