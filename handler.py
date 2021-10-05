"""Model handler for time series forecasting"""

import numpy as np
from jax import random

from numpyro.optim import Adam
from numpyro.infer import NUTS, MCMC, Trace_ELBO, SVI, Predictive

import matplotlib.pyplot as plt

class ModelHandler(object):
    def __init__(
        self, 
        model, 
        guide=None, 
        rng_seed=123,
        future=0,
        **data
    ) -> None:
        self.model = model
        self.guide = guide
        self.rng = random.PRNGKey(rng_seed)
        self.future = future
        self.data = data
        self.nuts = None
        self.nuts_samples = None
        self.svi = None
        self.svi_samples = None

    def run_nuts(self, n_warm=500, n_iter=1000, n_chains=1):
        assert self.nuts is None, "reset"
        nuts = NUTS(self.model)
        self.nuts = MCMC(nuts, num_warmup=n_warm, num_samples=n_iter, num_chains=n_chains)
        self.nuts.run(self.rng, extra_fields=('potential_energy', ), **self.data)
        print(f"Expected log density {np.mean(self.nuts.get_extra_fields()['potential_energy'])}")
        self.nuts.print_summary()
        predictive = Predictive(self.model, self.nuts.get_samples())
        self.nuts_samples = predictive(self.rng, **self.data, future=self.future)

    def run_svi(self, n_iter=1000, n_particles=10, step_size=.01, loss=Trace_ELBO, optim=Adam, n_samples=1000):
        assert self.svi is None, "reset"
        assert self.guide is not None
        self.svi = SVI(self.model, self.guide, optim(step_size), loss(n_particles))
        result = self.svi.run(self.rng, n_iter, **self.data)
        predictive = Predictive(self.model, guide=self.guide, params=result.params, num_samples=n_samples)
        self.svi_samples = predictive(self.rng, **self.data, future=self.future)

    def plot(self, y, months):
        assert self.nuts_samples is not None or self.svi_samples is not None
        if self.nuts_samples is not None:
            plt.figure(figsize=(20, 4))
            plt.plot(months, y, color='black')
            T = self.nuts_samples['obs'].shape[1]
            plt.plot(np.arange(T)+1, self.nuts_samples['obs'].mean(axis=0), color='blue')
            percentiles = np.percentile(self.nuts_samples['obs'], [5., 95.], axis=0)
            plt.fill_between(np.arange(T)+1, percentiles[0, :], percentiles[1, :], color='lightblue')
            plt.show()
        if self.svi_samples is not None:
            plt.figure(figsize=(20, 4))
            plt.plot(months, y, color='black')
            T = self.svi_samples['obs'].shape[1]
            plt.plot(np.arange(T)+1, self.svi_samples['obs'].mean(axis=0), color='blue')
            percentiles = np.percentile(self.svi_samples['obs'], [5., 95.], axis=0)
            plt.fill_between(np.arange(T)+1, percentiles[0, :], percentiles[1, :], color='lightblue')
            plt.show()