from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward
import tensorflow as tf
#from utils import policy, rollout, Normalised_Env
from gpflow import set_trainable
import gym
import numpy as np
import sys
import copy
import math
from gym import spaces, error
from skopt import Optimizer
from skopt.acquisition import gaussian_lcb
import matplotlib
import matplotlib.pyplot as plt
from gpflow import config
from gym import make
from matplotlib.ticker import MaxNLocator
float_type = config.default_float()


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, x, random)
            u = np.array(u)
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                if done: break
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            Y.append(np.array([x_new - x])) 
            ep_return_sampled += r 
            x = x_new
            if done: break
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full
    


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None,:])[0, :]

np.random.seed(0)

class dmboEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.reward_range = (0,1)
        # Action: set hyperparameter lambda of the lower confidence bound.
        self.action_space = spaces.Box(
            low=np.array([-2]), high=np.array([2]), dtype=np.float16)
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([0.0]),
            dtype=np.float16)
        
        #self.reset()
        

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        np.random.seed(0) #0
        acq_func_kwargs = {"kappa": 2}
        def objective(params):
            x0 = np.array([params[0], params[1], params[2],
                   params[3], params[4]])
            return sum(100.0*(x0[1:]-x0[:-1]**2.0)**2.0 + (1-x0[:-1])**2.0)
           
           
        paramSpace = [(-2.0, 2.0),
              (-2.0, 2.0),
              (-2.0, 2.0),
              (-2.0, 2.0),
              (-2.0, 2.0)]
        self.opt = Optimizer(paramSpace, "GP", n_initial_points=2,
                        acq_func="LCB", acq_optimizer="sampling",
                        acq_func_kwargs=acq_func_kwargs)
        self.opt.run(objective, n_iter=5)
        self.state = np.array([-min(self.opt.yi)/425])
        self.objective = objective
        self.done = False
        return self.state

    def render(self):
        return 1

    def step(self, action):
        """
        This method iterates DMBO forward for 4 iterations

        Parameters
        ----------
        action : float
            The action is the lambda value of the lower confidence bound.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        # test if action is valid
        assert self.action_space.contains(action)
        action = action + 2
        acq_func_kwargs = {"kappa": action}
        self.opt.acq_func_kwargs = acq_func_kwargs
        self.opt.update_next()
        self.opt.run(self.objective, n_iter=4)#4
        state_new = np.array([-min(self.opt.yi)/425])
        reward = np.array(np.exp(-(1/(2*0.2))*np.power((state_new-0.2),2)))
        done = False
        
        return state_new, reward, done, {}



env = dmboEnv()
    
SUBS = 1 
T = 20 # 20 
# Initial random rollouts to generate a dataset
X, Y, _, r = rollout(env=env, pilco=None, random=True, timesteps=T, SUBS = SUBS, render=False)#T
X = np.float64(X)
Y = np.float64(Y)
r = np.float64(r)
for i in range(1,5): #5
    X_, Y_, _, r_ = rollout(env=env, pilco=None, random=True,  timesteps=T, SUBS = SUBS, render=False)#T
    X = np.vstack((X, X_))
    Y = np.hstack((Y, Y_))
    r = np.vstack((r, r_))

plt.plot(r)
plt.show()

n = Y.size
Y = Y.reshape(n,1)

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim

controller = LinearController(state_dim=state_dim, control_dim=control_dim, max_action = 2)

target = np.array([0.0])
weights = np.diag([1.0])
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R)


for model in pilco.mgpr.models:#
    model.likelihood.variance.assign(0.05)
    set_trainable(model.likelihood.variance, False)


for rollouts in range(20):
    pilco.optimize_models(restarts = 2)
    pilco.optimize_policy(maxiter = 100, restarts = 3)
#    import pdb; pdb.set_trace()
    X_new, Y_new, _, r_ = rollout(env=env, pilco=pilco, timesteps=T, SUBS = SUBS, render=False)
    # Update dataset
    X_new = np.float64(X_new)
    Y_new = np.float64(Y_new)
    X = np.vstack((X, X_new))
    n = Y_new.size
    Y_new = Y_new.reshape(n,1)
    Y = np.vstack((Y, Y_new))
    r = np.vstack((r, r_))
    pilco.mgpr.set_data((X, Y))
    
plt.plot(r)
plt.xlabel('Episodes')  
plt.ylabel('Reward') 
plt.title("One Black-Box Function")
plt.savefig('dmboEx'+ 'One' +'.png', dpi=300)
plt.show()

l = X_new[:,1] + 2
plt.plot(l)
yint = []
locs, labels = plt.xticks()
for each in locs:
    yint.append(int(each))
plt.xticks(yint)
plt.xlim(xmin=1) #0
plt.xlim(xmax=T) #19
plt.xlabel('Timestep (one timestep equals four iterations of DMBO)')  
plt.ylabel('Lambda') 
plt.title("Last episode")
plt.savefig('dmboEx'+ 'OneLambda' +'.png', dpi=300)
plt.show()


