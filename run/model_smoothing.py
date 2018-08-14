import pdb
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..')
src_dir = os.path.join(this_dir, '..', 'src')
results_dir = os.path.join(this_dir, '..', 'analysis', 'results')
sys.path.append(pkg_dir)
sys.path.append(src_dir)

from src.policies.policies import model_smoothed_fitted_q
from src.Glucose import Glucose
from src.transition_estimation import MultivariateLinear
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime
import yaml
import multiprocessing as mp


def fitted_q_with_model_smoothing(index):
  np.random.seed(index)

  env = Glucose(horizon=1)
  gamma = 0.9
  number_of_value_iterations = 1
  transition_model_fitter = MultivariateLinear

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

    results['alpha_mb'].append(float(alpha_mb))
    results['mb_variances'].append(float(bootstrapped_mse_components_['mb_variances']))
    results['mf_variances'].append(float(bootstrapped_mse_components_['mf_variances']))
    results['mb_biases'].append(float(bootstrapped_mse_components_['mb_biases']))

  results['score'] = float(episode_score)

  return results


if __name__ == '__main__':
  n_replicate = 2
  pool = mp.Pool(int(np.min((n_replicate, int(np.floor(mp.cpu_count() / 2))))))
  replicates_results_list = pool.map(fitted_q_with_model_smoothing, range(n_replicate))

  # Collect results for each replicate into final results dictionary
  final_results = {}
  scores_list = []
  for i, dict_ in enumerate(replicates_results_list):
    final_results[i] = dict_
    scores_list.append(float(dict_['score']))
  final_results['summary'] = {'score mean': float(np.mean(scores_list)),
                              'score se': float(np.std(scores_list) / np.sqrt(len(scores_list))),
                              'score list': scores_list}

  # Write to file
  suffix = '.'.join([datetime.datetime.now().strftime("%y%m%d_%H%M%S"), 'yml'])
  basename = 'mse_smooth'
  fname = '_'.join([basename, suffix])
  fname = os.path.join(results_dir, fname)
  with open(fname, 'w') as outfile:
    yaml.dump(final_results, outfile)
