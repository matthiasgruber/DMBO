from pilco.models import PILCO
from pilco.controllers import LinearController
from pilco.rewards import ExponentialReward
from gpflow import set_trainable
import gym
import numpy as np
from gym import spaces
from skopt import Optimizer
import matplotlib.pyplot as plt

np.random.seed(0)

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
    
class dmboEnv1(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.reward_range = (0,200)
        # Action: set hyperparameter lambda of the lower confidence bound.
        self.action_space = spaces.Box(
            low=np.array([-2]), high=np.array([2]), dtype=np.float16)
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-1.5]), high=np.array([0.2]),
            dtype=np.float16)
        
        #self.reset()
        

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        np.random.seed(0)
        acq_func_kwargs = {"kappa": 2}
        def objective(params):
            x0 = np.array([params[0], params[1], params[2],
                           params[3], params[4], params[5],
                           params[6], params[7], params[8]])
            return sum(100.0*(x0[1:]-x0[:-1]**2.0)**2.0 + (1-x0[:-1])**2.0)
           
           
        paramSpace = [(-2.0, 2.0),(-2.0, 2.0),
                      (-2.0, 2.0),(-2.0, 2.0),
                      (-2.0, 2.0),(-2.0, 2.0),
                      (-2.0, 2.0),(-2.0, 2.0),(-2.0, 2.0)]
        self.opt = Optimizer(paramSpace, "GP", n_initial_points=4,
                        acq_func="LCB", acq_optimizer="sampling",
                        acq_func_kwargs=acq_func_kwargs)
        self.opt.run(objective, n_iter=5)
        self.state = np.array([(-min(self.opt.yi)/940.28)])
        self.objective = objective
        self.done = False
        return self.state

    def render(self):
        return 1

    def step(self, action):
        """
        This method iterates DMBO forward for 20 iterations

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
        action = action + 2.5
        acq_func_kwargs = {"kappa": action}
        self.opt.acq_func_kwargs = acq_func_kwargs
        self.opt.update_next()
        self.opt.run(self.objective, n_iter=4) # 5 ###############
        state_new = np.array([(-min(self.opt.yi)/940.28)]) #-np.array(min(self.opt.yi))/42
        reward = np.array(np.exp(-(1/(2*0.2))*np.power((state_new-0.2),2)))
        done = False
        
        return state_new, reward, done, {}
    
    
class dmboEnv2(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.reward_range = (0,200)
        # Action: set hyperparameter lambda of the lower confidence bound.
        self.action_space = spaces.Box(
            low=np.array([-2]), high=np.array([2]), dtype=np.float16)
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-1.5]), high=np.array([0.2]),
            dtype=np.float16)
        
        #self.reset()
        

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        np.random.seed(0)
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
        #random.seed(datetime.now())
        #self.opt.run(objective, n_iter=4)
        self.state = np.array([(-min(self.opt.yi)/425)]) #-np.array([min(self.opt.yi)])/425
        self.objective = objective
        self.done = False
        return self.state

    def render(self):
        return 1

    def step(self, action):
        """
        This method iterates DMBO forward for 20 iterations

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
        self.opt.run(self.objective, n_iter=4) #5 #########################
        state_new = np.array([(-min(self.opt.yi)/425)]) #-np.array(min(self.opt.yi))/425
        
        #state = np.array([min(self.opt.yi), len(self.opt.Xi)/10])
        #reward = (50-state[1])/10
        reward = np.array(np.exp(-(1/(2*0.2))*np.power((state_new-0.2),2)))
        done = False
        
        return state_new, reward, done, {}

env1 = dmboEnv1()
env2 = dmboEnv2()
    
SUBS = 1
T = 20 #10 #######################
# Initial random rollouts to generate a dataset
X, Y, _, r1 = rollout(env=env1, pilco=None, random=True, timesteps=T, SUBS = SUBS, render=False)
X_, Y_, _, r2 = rollout(env=env2, pilco=None, random=True,  timesteps=T, SUBS = SUBS, render=False)
X = np.vstack((X, X_))
Y = np.hstack((Y, Y_))
for i in range(1,5):
    X_, Y_, _, r_1 = rollout(env=env1, pilco=None, random=True,  timesteps=T, SUBS = SUBS, render=False)
    X = np.vstack((X, X_))
    Y = np.hstack((Y, Y_))
    r1 = np.vstack((r1, r_1))
for i in range(1,5):
    X_, Y_, _, r_2 = rollout(env=env2, pilco=None, random=True,  timesteps=T, SUBS = SUBS, render=False)
    X = np.vstack((X, X_))
    Y = np.hstack((Y, Y_))
    r2 = np.vstack((r2, r_2))

#plt.plot(r1)
#plt.plot(r2)

n = Y.size
Y = Y.reshape(n,1)

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim

#controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=25, max_action = 2)
controller = LinearController(state_dim=state_dim, control_dim=control_dim, max_action = 2)
                      
#R = LinearReward(state_dim=state_dim, W=1)

target = np.array([0.0])
weights = np.diag([1.0])
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)


pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R)


for model in pilco.mgpr.models:
    model.likelihood.variance.assign(0.1) #0.1
    set_trainable(model.likelihood.variance, False)


for rollouts in range(50): 
    pilco.optimize_models(restarts = 2)
    pilco.optimize_policy(maxiter = 50, restarts = 1)
#    import pdb; pdb.set_trace()
    X_new, Y_new, _, r_1 = rollout(env=env1, pilco=pilco, timesteps=T, SUBS = SUBS, render=False)
    # Update dataset
    X_new = np.float64(X_new)
    Y_new = np.float64(Y_new)
    X = np.vstack((X, X_new))
    n = Y_new.size
    Y_new = Y_new.reshape(n,1)
    Y = np.vstack((Y, Y_new))
    r1 = np.vstack((r1, r_1))
    pilco.mgpr.set_data((X, Y))
    #pilco.optimize_models(restarts = 2)
    #pilco.optimize_policy(maxiter = 50, restarts = 1)
    X_new, Y_new, _, r_2 = rollout(env=env2, pilco=pilco, timesteps=T, SUBS = SUBS, render=False)
    # Update dataset
    X_new = np.float64(X_new)
    Y_new = np.float64(Y_new)
    X = np.vstack((X, X_new))
    n = Y_new.size
    Y_new = Y_new.reshape(n,1)
    Y = np.vstack((Y, Y_new))
    r2 = np.vstack((r2, r_2))
    pilco.mgpr.set_data((X, Y))
    
plt.plot(r1)
plt.plot(r2)
plt.xlabel('Episodes')  
plt.ylabel('Reward') 
plt.title("Two Black-Box Functions")
plt.savefig('dmboEx'+ 'Two' +'.png', dpi=300)
plt.show()

