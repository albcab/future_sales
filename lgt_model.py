"""Local and Global trend model"""

from dataclasses import dataclass

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.contrib.control_flow import scan, cond

@dataclass
class Item:
    shop_id: int
    item_id: int
    data: jnp.ndarray #array of length 33 - start (or start_shop)
    start: int #start of life of item, if no item data => start of shop life
    stop: int #stop of life of shop
    level: float
    trend: float

def lgt_model(y, item_means, item_stds, shop_means, shop_stds, future=0):
    n_shop, T = shop_means.shape
    n_item, _ = item_means.shape
    plate_shops = numpyro.plate('s', n_shop, dim=-1)
    plate_items = numpyro.plate('i', n_item, dim=-2)

    df = numpyro.sample('df', dist.Uniform(2, 20))

    rho_a = numpyro.sample('rhoa', dist.HalfCauchy(1.))
    rho_b = numpyro.sample('rhob', dist.HalfCauchy(1.))
    lamda_a = numpyro.sample('lambdaa', dist.HalfCauchy(1.))
    lamda_b = numpyro.sample('lamdab', dist.HalfCauchy(1.))
    alpha_a = numpyro.sample('alphaa', dist.HalfCauchy(1.))
    alpha_b = numpyro.sample('alphab', dist.HalfCauchy(1.))
    beta_a = numpyro.sample('betaa', dist.HalfCauchy(1.))
    beta_b = numpyro.sample('betab', dist.HalfCauchy(1.))
    tau_a = numpyro.sample('taua', dist.HalfCauchy(1.))
    tau_b = numpyro.sample('taub', dist.HalfCauchy(1.))
    with plate_shops:
        rho = 1.5 * numpyro.sample('rho', dist.Beta(rho_a, rho_b)) - .5
        lamda = numpyro.sample('lamda', dist.Beta(lamda_a, lamda_b))
        alpha = numpyro.sample('alpha', dist.Beta(alpha_a, alpha_b))
        beta = numpyro.sample('beta', dist.Beta(beta_a, beta_b))
        tau = numpyro.sample('tau', dist.Beta(tau_a, tau_b))

    cauchy_sd = jnp.max(item_means + 3 * item_stds**2, axis=1)
    gamma_mu = numpyro.sample('gammamu', dist.Normal(0., 1.))
    with plate_items:
        xi = numpyro.sample('xi', dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd))
        sigma = numpyro.sample('sigma', dist.HalfCauchy(cauchy_sd))
        gamma = numpyro.sample('gamma', dist.Cauchy(gamma_mu, cauchy_sd))

    def _transition(y, t):
        def _sub_transition(yt):
            y, t = yt

            y_hat = y.level + gamma[y.item_id] + y.level**rho[y.shop_id] + lamda[y.shop_id] * y.trend
            y_hat = jnp.clip(y_hat, a_min=0.)
            #use expected value when forecasting
            y_t = jnp.where(
                t > T,
                y_hat,
                jnp.where(
                    not jnp.isnan(y.data[t-y.start]),
                    y.data[t-y.start],
                    #use bayesian imputation if cnts missing, using item data or (of missing too) shop data
                    jnp.where(
                        not jnp.isnan(item_means[y.item_id, t]),
                        numpyro.sample(
                            f"cnt_impute_{y.shop_id}_{y.item_id}_{t}",
                            dist.Normal(item_means[y.item_id, t], item_stds[y.item_id, t])
                        ).mask(False),
                        numpyro.sample(
                            f"cnt_impute_{y.shop_id}_{y.item_id}_{t}",
                            dist.Normal(shop_means[y.shop_id, t], shop_stds[y.shop_id, t])
                        ).mask(False)
                    )
                )
            )
            _level = y.level
            y.level = alpha[y.shop_id] * y_t + (1-alpha[y.shop_id]) * _level
            y.level = jnp.clip(y.level, a_min=0.)

            y.trend = beta[y.shop_id] * (y.level-_level) + (1-beta[y.shop_id]) * y.trend

            sigma_hat = sigma[y.item_id] * y_hat**tau[y.shop_id] + xi[y.item_id]

            y_ = jnp.where(
                t > T,
                numpyro.sample('obs', dist.StudentT(df, y_hat, sigma_hat), obs=y_t),
                numpyro.sample('forecast', dist.StudentT(df, y_hat, sigma_hat))
            )
            return y, y_

        def _null_transition(yt):
            y, _ = yt
            return y, jnp.nan
        
        return cond(t > y.start and t < y.stop, _sub_transition, _null_transition, (y, t))

    _, Y = scan(_transition, y, jnp.arange(1, T+future))

def model(y, future=0):
    T = len(y)
    cauchy_sd = jnp.max(y) / 150

    df = numpyro.sample('df', dist.Uniform(2, 20))
    rho = 1.5 * numpyro.sample('rho', dist.Beta(1., 1.)) - .5
    lamda = numpyro.sample('lamda', dist.Beta(1, 1))
    alpha = numpyro.sample('alpha', dist.Beta(1, 1))
    beta = numpyro.sample('beta', dist.Beta(1, 1))
    tau = numpyro.sample('tau', dist.Beta(1,1))
    xi = numpyro.sample('xi', dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd))
    sigma = numpyro.sample('sigma', dist.HalfCauchy(cauchy_sd))
    gamma = numpyro.sample('gamma', dist.Cauchy(0, cauchy_sd))

    def _transition(carry, i):
        level, trend = carry

        y_hat = level + gamma * level**rho + lamda * trend
        y_hat = jnp.clip(y_hat, a_min=0.)
        #use expected value when forecasting
        y_i = jnp.where(i > T, y_hat, y[i])

        _level = level
        level = alpha * y_i + (1-alpha) * _level
        level = jnp.clip(level, a_min=0.)

        trend = beta * (level-_level) + (1-beta) * trend

        sigma_hat = sigma * y_hat**tau + xi

        y_ = numpyro.sample('obs', dist.StudentT(df, y_hat, sigma_hat))
        return (level, trend), y_

    init = y[0]
    with numpyro.handlers.condition(data={'obs': y[1:]}):
        _, Y = scan(_transition, (init, init), jnp.arange(1, T+future))
    
def guide(y, future=0):
    # numpyro.sample('df', dist.Uniform(2, 20))
    with numpyro.handlers.reparam(config={'df': TransformReparam()}):
        numpyro.sample(
            'df',
            dist.TransformedDistribution(
                dist.Beta(numpyro.param('adf', 1., constraint=dist.constraints.positive),
                          numpyro.param('bdf', 1., constriant=dist.constraints.positive)),
                dist.transforms.AffineTransform(2., 18.)
            )
        )
    numpyro.sample(
        'rho', 
        dist.Beta(numpyro.param('arho', 1., constraint=dist.constraints.positive), 
                  numpyro.param('brho', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'lamda', 
        dist.Beta(numpyro.param('alamda', 1., constraint=dist.constraints.positive), 
                  numpyro.param('blamda', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'alpha', 
        dist.Beta(numpyro.param('aalpha', 1., constraint=dist.constraints.positive), 
                  numpyro.param('balpha', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'beta', 
        dist.Beta(numpyro.param('abeta', 1., constraint=dist.constraints.positive), 
                  numpyro.param('bbeta', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'tau', 
        dist.Beta(numpyro.param('atau', 1., constraint=dist.constraints.positive), 
                  numpyro.param('btau', 1., constrain=dist.constraints.positive))
    )
    # numpyro.sample(
    #     'xi', 
    #     dist.TruncatedCauchy(low=1e-10, 
    #                          loc=numpyro.param('lxi', 1e-10, constraint=dist.constraints.greater_than(1e-10)), 
    #                          scale=numpyro.param('sxi', 1., constraint=dist.constraints.positive))
    # )
    numpyro.sample(
        'xi',
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('lxi', 0.), 
                        numpyro.param('sxi', 1., constraint=dist.constraints.positive)),
            dist.transforms.ExpTransform()
        )
    )
    # numpyro.sample(
    #     'sigma', 
    #     dist.HalfCauchy(numpyro.param('ssd', 1., constraint=dist.constraints.positive))
    # )
    numpyro.sample(
        'sigma',
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('msigma', 0.),
                        numpyro.param('ssigma', 1., constraint=dist.constraints.positive)),
            dist.transforms.ExpTransform()
        )
    )
    # numpyro.sample(
    #     'gamma', 
    #     dist.Cauchy(numpyro.param('mgamma', 0.),
    #                 numpyro.param('sgamma', 1., constraint=dist.constraints.positive))
    # )
    numpyro.sample(
        'gamma',
        dist.Normal(numpyro.param('mgamma', 0.),
                    numpyro.param('sgamma', 1., constraint=dist.constraints.positive))
    )