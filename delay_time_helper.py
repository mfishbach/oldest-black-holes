import numpy as np

import jax.numpy as jnp

from gw_pop_numpyro import config

#tmin, tmax in Myr

def ptau(taus, tmin, alpha = -1, tmax = config.cosmo_dict["age_of_universe"]):
    if alpha == -1:
        return jnp.where((taus >= tmin) & (taus <= tmax), taus**-1/(jnp.log(tmax) - jnp.log(tmin)), 0)
    else:
        return jnp.where((taus >= tmin) & (taus <= tmax), (alpha + 1) * taus**alpha/(tmax**(alpha + 1) - tmin**(alpha + 1)), 0)

def ptau_cumulative(taus, tmin, alpha = -1, tmax = config.cosmo_dict["age_of_universe"]):
    if alpha == - 1:
        return jnp.where( (taus >= tmin) & (taus <= tmax), (jnp.log(taus) - jnp.log(tmin)) / (jnp.log(tmax) - jnp.log(tmin)), 0)
    else:
        return jnp.where( (taus >= tmin) & (taus <= tmax), (taus**(alpha + 1) - tmin**(alpha + 1)) / (tmax**(alpha + 1) - tmin**(alpha + 1)), 0)

def inverse_cdf_powerlaw(quantile, tmin, alpha, tmax):
    if alpha == -1:
        logt = jnp.log(tmin) + (jnp.log(tmax) - jnp.log(tmin)) * quantile
        t = jnp.exp(logt)
    else:
        t = (quantile*(tmax**(alpha+1) - tmin**(alpha+1)) + tmin**(alpha+1))**(1.0/(alpha+1))
    return t

def draw_delay_times(tmin, shape, alpha = -1, tmax = config.cosmo_dict["age_of_universe"], seed = None):
    rng = np.random.default_rng(seed = seed)
    u = jnp.array(rng.random(shape))
    t = inverse_cdf_powerlaw(u, tmin, alpha, tmax)
    return t

def calculate_formation_redshift(samples, tmin, alpha = -1, seed = None):

    tL_merge = config.cosmo_dict["lookback_time"](samples["redshift"])

    tmax = config.cosmo_dict["lookback_time"](config.zmax) - tL_merge

    td = draw_delay_times(tmin, tL_merge.shape, alpha, tmax, seed)

    formation_lookback_time = tL_merge + td

    formation_redshift = config.cosmo_dict["z_at_lookback_time"](formation_lookback_time)

    return formation_redshift

def add_formation_tL(samples, tmin, alpha = -1, seed = None):

    tL_merge = config.cosmo_dict["lookback_time"](samples["redshift"])

    tmax = config.cosmo_dict["lookback_time"](config.zmax) - tL_merge

    td = draw_delay_times(tmin, tL_merge.shape, alpha, tmax, seed)

    samples["formation_lookback_time"] = tL_merge + td

    samples["tau_grid"] = jnp.logspace(jnp.log10(tmin), jnp.log10(config.cosmo_dict["lookback_time"](config.zmax)), 1000)

    samples["ptau_grid"] = ptau(samples["tau_grid"], tmin, alpha, config.cosmo_dict["lookback_time"](config.zmax))

def merger_rate_at_age_from_formation_delay(age_m_grid, tau_grid, ptau_grid, formation_rate_at_age_func):
    '''
    age_m_grid: 1-d array of times at which to evaluate merger rate
    tau_grid: 1-d array of delay times used in integration
    ptau_grid: 1-d array of delay time pdf evaluated at tau_grid, same shape as tau_grid
    formation_rate_at_age_func: function that gives formation rate at a given time (age)
    returns: merger rate evaluated at age_m_grid
    '''

    age_m_2d = age_m_grid[:, jnp.newaxis] #time at merger, elevate to a 2-d array

    age_f_2d = age_m_2d - tau_grid #time at formation, has shape (len(age_m_grid), len(tau_grid))

    formation_rate_2d = formation_rate_at_age_func(age_f_2d) #formation rate at formation time

    merger_rate = jnp.trapz(formation_rate_2d * ptau_grid, tau_grid, axis = -1) #integrate over delay time distribution

    return merger_rate
