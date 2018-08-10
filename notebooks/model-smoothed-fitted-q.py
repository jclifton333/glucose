
# coding: utf-8

# In[1]:


import sys
#sys.path.append('/mnt/c/Users/Jesse/Desktop/glucose')
#sys.path.append('/mnt/c/Users/Jesse/Desktop/glucose/src')
#sys.path.append('/mnt/c/Users/Jesse/Desktop/glucose/src/policies')
sys.path.append('/users/jclih/glucose')
sys.path.append('/users/jclih/glucose/src')
sys.path.append('/users/jclih/glucose/policies')


# In[2]:


from policies.policies import model_smoothed_fitted_q
from Glucose import Glucose
from transition_estimation import MultivariateLinear
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


env = Glucose(horizon=50)
transition_model_fitter = MultivariateLinear
pairwise_kernels_ = np.zeros((0,0))
kernel_sums = 0 
gamma = 0.9
number_of_value_iterations = 1


# In[ ]:


done = False
env.reset()
while not done:
    action, pairwise_kernel_, kernel_sums =         model_smoothed_fitted_q(env, gamma, RandomForestRegressor, number_of_value_iterations, transition_model_fitter,
                                pairwise_kernels_, kernel_sums)
    _, r, done = env.step(action)

