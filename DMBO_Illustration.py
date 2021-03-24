# based on: https://scikit-optimize.github.io/stable/auto_examples/exploration-vs-exploitation.html

import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt.acquisition import gaussian_lcb

def plot_optimizer(opt, x, fx, kappa, i):
    model = opt.models[-1]
    x_model = opt.space.transform(x.tolist())

    # Plot true function
    plt.plot(x, fx, "r--", label="True (unknown)")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([fx - 1.9600 * noise_level,
                             fx[::-1] + 1.9600 * noise_level]),
             alpha=.2, fc="r", ec="None")

    # Plot Model(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    plt.plot(x, y_pred, "g--", label=r"$\mu(x)$")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.2, fc="g", ec="None")

    # Plot sampled points
    plt.plot(opt.Xi, opt.yi,
             "r.", markersize=8, label="Observations")

    acq = gaussian_lcb(x_model, model, kappa)
    # shift down to make a better plot
    acq = 0.25 * acq - 2
    plt.plot(x, acq, "b", label="LCB(x)")

    # Adjust plot layout
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('x')  
    plt.ylabel('y')
    plt.title("Surrogate: GP; Infill Criterion: LCB; Lambda: " + str(kappa))
    plt.ylim((-4,4))
    plt.savefig('dmboEx'+ str(i) +'.png', dpi=300) 
    plt.show()

noise_level = 0.1

# Our 1D toy problem, this is the function we are trying to
# minimize
def objective(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) +\
           np.random.randn() * noise_level
           
###############################################

# Plot true function
plt.set_cmap("viridis")
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = np.array([objective(x_i, noise_level=0.0) for x_i in x])
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")
plt.legend()
plt.grid()
plt.xlabel('x')  
plt.ylabel('y') 
plt.title("Black-Box Function")
plt.ylim((-4,4))
i = 1
plt.savefig('dmboEx'+ str(i) +'.png', dpi=300)
plt.show()

#############################################

acq_func_kwargs = {"kappa": 0}

opt = Optimizer([(-2.0, 2.0)], "GP", n_initial_points=2,
                acq_func="LCB", acq_optimizer="sampling",
                acq_func_kwargs=acq_func_kwargs)

opt.run(objective, n_iter=20)
kappa = 0
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = np.array([objective(x_i, noise_level = 0.0) for x_i in x])
plot_optimizer(opt, x, fx, kappa, i = 2)

###############################################

acq_func_kwargs = {"kappa": 5}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 5
plot_optimizer(opt, x, fx, kappa, i = 3)

###############################################

acq_func_kwargs = {"kappa": 5}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 5
plot_optimizer(opt, x, fx, kappa, i = 4)
###############################################

acq_func_kwargs = {"kappa": 5}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 5
plot_optimizer(opt, x, fx, kappa, i = 5)
###############################################

acq_func_kwargs = {"kappa": 5}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 5
plot_optimizer(opt, x, fx, kappa, i = 6)
###############################################

acq_func_kwargs = {"kappa": 5}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 5
plot_optimizer(opt, x, fx, kappa, i = 7)
###############################################

acq_func_kwargs = {"kappa": 1}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 1
plot_optimizer(opt, x, fx, kappa, i = 8)
###############################################

acq_func_kwargs = {"kappa": 1}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 1
plot_optimizer(opt, x, fx, kappa, i = 9)
###############################################

acq_func_kwargs = {"kappa": 1}
opt.acq_func_kwargs = acq_func_kwargs
opt.update_next()
opt.run(objective, n_iter=1)
kappa = 1
plot_optimizer(opt, x, fx, kappa, i = 10)
