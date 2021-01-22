"""
This code implements the different utility functions of exotic option pricing
Author:     Yufei Shen
Date:       1/1/2021
# pricing-lib

"""

import numpy as np
from scipy.stats import norm
from Utils.other_utils import dictGetAttr, frequency_counts_dict
from Utils.vanilla_utils import BSOpt, single_asset_vol_base
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GBM_barrier_obj(single_asset_vol_base):
    """Pricing object for GBM dynamics under barrier options"""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        barrier,
        prod_type="EUROPEAN BARRIER",
        **kwargs
    ):
        super().__init__(
            isCall, spot, strike, tau, ir, dividend_yield, prod_type, **kwargs
        )
        self.isContinuous = isContinuous
        self.flavor = flavor
        self.barrier = barrier
        self.vol = dictGetAttr(self.other_params, "vol", None)
        if not self.isContinuous:
            self.m_intervals = dictGetAttr(self.other_params, "m_intervals", None)
            if not self.m_intervals:
                self.barrier_obs_freq = dictGetAttr(
                    self.other_params, "barrier_obs_freq", None
                ).upper()
                if not self.barrier_obs_freq:
                    raise ValueError(
                        "Need to provide observation frequency for discrete barrier options!"
                    )
                self.m_intervals = (
                    self.tau * frequency_counts_dict[self.barrier_obs_freq]
                )

    def price(self):
        if self.isContinuous:
            if self.flavor == "DOWN-AND-IN":
                return single_barrier_dni(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier,
                    self.ir,
                    self.dividend_yield,
                )
            elif self.flavor == "DOWN-AND-OUT":
                return single_barrier_dno(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier,
                    self.ir,
                    self.dividend_yield,
                )
            elif self.flavor == "UP-AND-IN":
                return single_barrier_uni(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier,
                    self.ir,
                    self.dividend_yield,
                )
            elif self.flavor == "UP-AND-OUT":
                return single_barrier_uno(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier,
                    self.ir,
                    self.dividend_yield,
                )
        else:  # If barrier is NOT continuous observed
            if self.flavor == "DOWN-AND-IN":
                return single_barrier_dni(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier
                    * np.exp(-0.5826 * self.vol * np.sqrt(self.tau / self.m_intervals)),
                    self.ir,
                    self.dividend_yield,
                )
            elif self.flavor == "DOWN-AND-OUT":
                return single_barrier_dno(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier
                    * np.exp(-0.5826 * self.vol * np.sqrt(self.tau / self.m_intervals)),
                    self.ir,
                    self.dividend_yield,
                )
            elif self.flavor == "UP-AND-IN":
                return single_barrier_uni(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier
                    * np.exp(0.5826 * self.vol * np.sqrt(self.tau / self.m_intervals)),
                    self.ir,
                    self.dividend_yield,
                )
            elif self.flavor == "UP-AND-OUT":
                return single_barrier_uno(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.vol,
                    self.tau,
                    self.barrier
                    * np.exp(0.5826 * self.vol * np.sqrt(self.tau / self.m_intervals)),
                    self.ir,
                    self.dividend_yield,
                )


def barrier_lamda(vol, r, q):
    return (r - q + vol ** 2 / 2) / (vol ** 2)


def barrier_y(spot, strike, vol, tau, barrier, r, q):
    term1 = np.log(barrier ** 2 / spot / strike) / (vol * np.sqrt(tau))
    term2 = barrier_lamda(vol, r, q) * vol * np.sqrt(tau)
    return term1 + term2


def barrier_x1(spot, vol, tau, barrier, r, q):
    term1 = np.log(spot / barrier) / (vol * np.sqrt(tau))
    term2 = barrier_lamda(vol, r, q) * vol * np.sqrt(tau)
    return term1 + term2


def barrier_y1(spot, vol, tau, barrier, r, q):
    term1 = np.log(barrier / spot) / (vol * np.sqrt(tau))
    term2 = barrier_lamda(vol, r, q) * vol * np.sqrt(tau)
    return term1 + term2


def single_barrier_dni(isCall, spot, strike, vol, tau, barrier, r, q):
    """Single barrier down-and-in option"""
    if isCall:
        if barrier <= strike:
            term1 = (
                spot
                * np.exp(-q * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q))
                * norm.cdf(barrier_y(spot, strike, vol, tau, barrier, r, q))
            )
            term2 = (
                strike
                * np.exp(-r * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q) - 2)
                * norm.cdf(
                    barrier_y(spot, strike, vol, tau, barrier, r, q)
                    - vol * np.sqrt(tau)
                )
            )
            return term1 - term2
        else:
            vanilla_call = BSOpt(isCall, spot, strike, vol, tau, r, q)
            term1 = (
                spot
                * np.exp(-q * tau)
                * norm.cdf(barrier_x1(spot, vol, tau, barrier, r, q))
            )
            term2 = (
                strike
                * np.exp(-r * tau)
                * norm.cdf(
                    barrier_x1(spot, vol, tau, barrier, r, q) - vol * np.sqrt(tau)
                )
            )
            term3 = (
                spot
                * np.exp(-q * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q))
                * norm.cdf(barrier_y1(spot, vol, tau, barrier, r, q))
            )
            term4 = (
                strike
                * np.exp(-r * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q) - 2)
                * norm.cdf(
                    barrier_y1(spot, vol, tau, barrier, r, q) - vol * np.sqrt(tau)
                )
            )
            return vanilla_call - (term1 - term2 - term3 + term4)
    else:
        if barrier <= strike:
            term1 = (
                spot
                * np.exp(-q * tau)
                * norm.cdf(-barrier_x1(spot, vol, tau, barrier, r, q))
            )
            term2 = (
                strike
                * np.exp(-r * tau)
                * norm.cdf(
                    -barrier_x1(spot, vol, tau, barrier, r, q) + vol * np.sqrt(tau)
                )
            )
            term3 = (
                spot
                * np.exp(-q * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q))
                * (
                    norm.cdf(barrier_y(spot, strike, vol, tau, barrier, r, q))
                    - norm.cdf(barrier_y1(spot, vol, tau, barrier, r, q))
                )
            )
            term4 = (
                strike
                * np.exp(-r * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q) - 2)
                * (
                    norm.cdf(
                        barrier_y(spot, strike, vol, tau, barrier, r, q)
                        - vol * np.sqrt(tau)
                    )
                    - norm.cdf(
                        barrier_y1(spot, vol, tau, barrier, r, q) - vol * np.sqrt(tau)
                    )
                )
            )
            return -term1 + term2 + term3 - term4
        else:
            return BSOpt(isCall, spot, strike, vol, tau, r, q)


def single_barrier_dno(isCall, spot, strike, vol, tau, barrier, r, q):
    """Single barrier down-and-out option
    Use the in-out parity"""
    vanilla_price = BSOpt(isCall, spot, strike, vol, tau, r, q)
    return vanilla_price - single_barrier_dni(
        isCall, spot, strike, vol, tau, barrier, r, q
    )


def single_barrier_uni(isCall, spot, strike, vol, tau, barrier, r, q):
    """Single barrier up-and-in option"""
    if isCall:
        if barrier <= strike:
            return BSOpt(isCall, spot, strike, vol, tau, r, q)
        else:
            term1 = (
                spot
                * np.exp(-q * tau)
                * norm.cdf(barrier_x1(spot, vol, tau, barrier, r, q))
            )
            term2 = (
                strike
                * np.exp(-r * tau)
                * norm.cdf(
                    barrier_x1(spot, vol, tau, barrier, r, q) - vol * np.sqrt(tau)
                )
            )
            term3 = (
                spot
                * np.exp(-q * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q))
                * (
                    norm.cdf(-barrier_y(spot, strike, vol, tau, barrier, r, q))
                    - norm.cdf(-barrier_y1(spot, vol, tau, barrier, r, q))
                )
            )
            term4 = (
                strike
                * np.exp(-r * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q) - 2)
                * (
                    norm.cdf(
                        -barrier_y(spot, strike, vol, tau, barrier, r, q)
                        + vol * np.sqrt(tau)
                    )
                    - norm.cdf(
                        -barrier_y1(spot, vol, tau, barrier, r, q) + vol * np.sqrt(tau)
                    )
                )
            )
            return term1 - term2 - term3 + term4
    else:
        if barrier <= strike:
            vanilla_put = BSOpt(isCall, spot, strike, vol, tau, r, q)
            term1 = (
                spot
                * np.exp(-q * tau)
                * norm.cdf(-barrier_x1(spot, vol, tau, barrier, r, q))
            )
            term2 = (
                strike
                * np.exp(-r * tau)
                * norm.cdf(
                    -barrier_x1(spot, vol, tau, barrier, r, q) + vol * np.sqrt(tau)
                )
            )
            term3 = (
                spot
                * np.exp(-q * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q))
                * norm.cdf(-barrier_y1(spot, strike, vol, tau, barrier, r, q))
            )
            term4 = (
                strike
                * np.exp(-r * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q) - 2)
                * norm.cdf(
                    -barrier_y1(spot, strike, vol, tau, barrier, r, q)
                    + vol * np.sqrt(tau)
                )
            )
            return vanilla_put - (-term1 + term2 + term3 - term4)
        else:
            term1 = (
                spot
                * np.exp(-q * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q))
                * norm.cdf(-barrier_y(spot, strike, vol, tau, barrier, r, q))
            )
            term2 = (
                strike
                * np.exp(-r * tau)
                * (barrier / spot) ** (2 * barrier_lamda(vol, r, q) - 2)
                * norm.cdf(
                    -barrier_y(spot, strike, vol, tau, barrier, r, q)
                    + vol * np.sqrt(tau)
                )
            )
            return -term1 + term2


def single_barrier_uno(isCall, spot, strike, vol, tau, barrier, r, q):
    """Single barrier up-and-out option
    Use in-out parity"""
    vanilla_price = BSOpt(isCall, spot, strike, vol, tau, r, q)
    return vanilla_price - single_barrier_uni(
        isCall, spot, strike, vol, tau, barrier, r, q
    )
