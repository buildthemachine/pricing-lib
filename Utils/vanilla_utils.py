"""
This code implements the different utility functions of option pricing
Author:     Yufei Shen
Date:       12/21/2020
# pricing-lib

"""

import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.sparse
from scipy.stats import norm, ncx2
from abc import ABCMeta, abstractmethod
from LocalVol.lv_utils import dictGetAttr
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class vanilla_vol_base(metaclass=ABCMeta):
    """Base volatility class for vanilla products
    Supports two functions: price and impliedVol"""

    @abstractmethod
    def __init__(self, isCall, x0, strike, tau, **kwargs):
        self.isCall = isCall
        self.x0 = x0  # x0 can be either forward or spot prices
        self.strike = strike
        self.tau = tau
        kwargs = {
            k.upper(): v for k, v in kwargs.items()
        }  # Convert all keys to upper case.
        self.other_params = kwargs

    @abstractmethod
    def price(self):
        raise NotImplementedError

    def impliedVol(self):
        target = self.price()
        return BSImpVol(
            self.isCall,
            self.x0,
            self.strike,
            self.tau,
            target,
            self.ir,
            self.dividend_yield,
        )


class Bachlier_obj(vanilla_vol_base):
    """Pricing object for Bachelier model"""

    def __init__(self, isCall, spot, strike, tau, **kwargs):
        super().__init__(isCall, spot, strike, tau, **kwargs)
        self.vol = dictGetAttr(self.other_params, "vol", None)
        self.ir = dictGetAttr(self.other_params, "ir", None)
        self.dividend_yield = dictGetAttr(self.other_params, "dividend_yield", None)

    def price(self):
        return BachlierSpot(
            self.isCall, self.x0, self.strike, self.vol, self.ir, self.tau
        )


class CEV_obj(vanilla_vol_base):
    """Pricing object for CEV model"""

    def __init__(self, isCall, spot, strike, tau, **kwargs):
        super().__init__(isCall, spot, strike, tau, **kwargs)
        self.lamda = dictGetAttr(self.other_params, "lamda", None)
        self.p = dictGetAttr(self.other_params, "p", None)
        self.ir = dictGetAttr(self.other_params, "ir", None)
        self.dividend_yield = dictGetAttr(self.other_params, "dividend_yield", None)

    def price(self):
        return CEVSpot(
            self.isCall,
            self.x0,
            self.strike,
            self.tau,
            self.lamda,
            self.p,
            self.ir,
            self.dividend_yield,
        )


class BS_obj(vanilla_vol_base):
    """Pricing object for Black-Scholes equation"""

    def __init__(self, isCall, spot, strike, tau, **kwargs):
        super().__init__(isCall, spot, strike, tau, **kwargs)
        self.vol = dictGetAttr(self.other_params, "vol", None)
        self.ir = dictGetAttr(self.other_params, "ir", None)
        self.dividend_yield = dictGetAttr(self.other_params, "dividend_yield", None)

    def price(self):
        return BSOpt(
            self.isCall,
            self.x0,
            self.strike,
            self.vol,
            self.tau,
            self.ir,
            self.dividend_yield,
        )


class sabr_black_vol(vanilla_vol_base):
    """TODO: Writing unit test files for sabr
    SABR model does not depend on interest rate, therefore set r=q=0"""

    def __init__(self, isCall, x0, strike, tau, sabr_params, eps=1e-6):
        """This methods provides an interface for constructing a vol smile.
        The strike k provided can be a vector."""
        super().__init__(isCall, x0, strike, tau)
        self.sabr_params = sabr_params
        # If f is too close to k, the sabr formular will have zero on denominator.
        # eps is a threshold for the minimum RELATIVE difference between f and k.
        self.eps = eps

    def impliedVol(self):
        sig0, beta, rho, alpha = self.sabr_params
        if (abs(alpha)) < self.eps:
            alpha = self.eps
        f = self.x0
        k = self.strike

        def z_func(alpha, sig0, f, k, beta):
            return (alpha / sig0) * (f * k) ** ((1 - beta) / 2) * np.log(f / k)

        def x(z, rho):
            return np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        if np.min(np.fabs(f / k - 1)) < self.eps:
            k = k * (1 + self.eps)
        z = z_func(alpha, sig0, f, k, beta)
        num1 = sig0
        den1 = (f * k) ** ((1 - beta) / 2) * (
            1
            + (1 - beta) ** 2 / 24 * (np.log(f / k)) ** 2
            + (1 - beta) ** 4 / 1920 * (np.log(f / k)) ** 4
        )
        num2 = z
        den2 = x(z, rho)
        term3 = 1 + self.tau * (
            (1 - beta) ** 2 / 24 * sig0 ** 2 / (f * k) ** (1 - beta)
            + 1 / 4 * rho * beta * alpha * sig0 / (f * k) ** ((1 - beta) / 2)
            + (2 - 3 * rho ** 2) / 24 * alpha ** 2
        )
        #         print(f"Term is {tex} and term3 is {term3}")
        #         pdb.set_trace()
        return num1 / den1 * num2 / den2 * term3

    def price(self):
        impVol = self.impliedVol()
        return BSOpt(self.isCall, self.x0, self.strike, impVol, self.tau, 0, 0)

    def calibrate(self):
        """TODO: Placeholder for calibration routine"""
        pass


def d1(spot, strike, vol, tau, r, q):
    """The d1 function in Black Scholes"""
    return (np.log(spot / strike) + (r - q + 0.5 * vol ** 2) * tau) / vol / np.sqrt(tau)


def d2(spot, strike, vol, tau, r, q):
    """The d1 function in Black Scholes"""
    return (np.log(spot / strike) + (r - q - 0.5 * vol ** 2) * tau) / vol / np.sqrt(tau)


def fd1(fwd, strike, vol, tau):
    """The d1 function in black sholes with forwards"""
    return (np.log(fwd / strike) + 0.5 * vol ** 2 * tau) / vol / np.sqrt(tau)


def fd2(fwd, strike, vol, tau):
    """The d1 function in black sholes with forwards"""
    return (np.log(fwd / strike) - 0.5 * vol ** 2 * tau) / vol / np.sqrt(tau)


def d_Bache(fwd, strike, vol, tau):
    """This is the 'd' term in the Bachelier pricing formula"""
    return (fwd - strike) / vol / np.sqrt(tau)


def BSOpt(isCall, spot, strike, vol, tau, r, q):
    """
    r: risk free interest rate
    q: convenience yield (-) or dividend payment (+)
    """
    if isCall:
        return spot * np.exp(-q * tau) * norm.cdf(
            d1(spot, strike, vol, tau, r, q)
        ) - strike * np.exp(-r * tau) * norm.cdf(d2(spot, strike, vol, tau, r, q))
    else:
        return strike * np.exp(-r * tau) * norm.cdf(
            -d2(spot, strike, vol, tau, r, q)
        ) - spot * np.exp(-q * tau) * norm.cdf(-d1(spot, strike, vol, tau, r, q))


def undiscBSOptFwd(isCall, fwd, strike, vol, tau):
    """This is the Black's option pricing formula for fowards"""

    # if strike < 1.0e-12 * fwd:
    #     if isCall:
    #         return fwd - strike
    #     else:
    #         return 0.0

    if isCall:
        return fwd * norm.cdf(fd1(fwd, strike, vol, tau)) - strike * norm.cdf(
            fd2(fwd, strike, vol, tau)
        )
    else:
        return strike * norm.cdf(-fd2(fwd, strike, vol, tau)) - fwd * norm.cdf(
            -fd1(fwd, strike, vol, tau)
        )


def undiscBachelierFwd(isCall, fwd, strike, vol, tau):
    """This module implementes Bachelier's formula for normal volatility option pricing. The normal dynamics is given by:
    dS(t)=lambda dW(t)
    where lambda is the volatilty
    """
    d = d_Bache(fwd, strike, vol, tau)
    callPrice = (fwd - strike) * norm.cdf(d) + vol * np.sqrt(tau) * norm.pdf(d)
    if isCall:
        return callPrice
    else:
        return callPrice - fwd + strike  # From put-call parity


def BachlierSpot(isCall, spot, strike, vol, ir, tau):
    """This module implementes Bachelier's formula for normal volatility option pricing. The normal dynamics is given by:
    dS(t)=rS(t)dt + lambda dW(t)
    where lambda is the volatilty
    """
    df = np.exp(-ir * tau)
    d = d_Bache(spot / df, strike, vol, tau)
    callPrice = (spot / df - strike) * norm.cdf(d) + vol * np.sqrt(tau) * norm.pdf(d)
    callPrice = callPrice * df
    if isCall:
        return callPrice
    else:
        return callPrice - spot + strike * df


def BlackImpVol(isCall, discount, fwd, strike, tau, price):
    """Back out the implied volatility from Black's model, through a root finding approach
    For undiscounted pricing as input, set discount = 1."""

    def loss(vol):
        return undiscBSOptFwd(isCall, fwd, strike, vol, tau) - price / discount

    volMin = 1.0e-6
    volMax = 10
    Vol = scipy.optimize.brenth(loss, volMin, volMax)
    return Vol


def BSImpVol(isCall, spot, strike, tau, price, r, q):
    """Back out the implied volatility from Black-Scholes model, through a root finding approach
    For undiscounted pricing as input, set r=q=0.
    r: risk-free rate
    q: dividend yield
    """

    def loss(vol):
        return BSOpt(isCall, spot, strike, vol, tau, r, q) - price

    volMin = 1.0e-6
    volMax = 10
    Vol = scipy.optimize.brenth(loss, volMin, volMax)
    return Vol


def undiscCEVFwd(isCall, fwd, strike, lamda, tau, p):
    """The CEV model is given by equation 7.9 in Andersen. In particular:
    dS(t)=lambda S(t)^p dW(t)
    The pricing formula is given by Prop. 7.2.6 in Andersen.
    Note that this formula assumes absorbing boundary at fwd=0, thus the p=0 case is
    slightly different from Bachelier. On the other hand, when p->1, the CEV price
    converge to the Black formula as fwd=0 is not attainable under lognormal dynamics.
    """
    a = strike ** (2 * (1 - p)) / ((1 - p) ** 2 * lamda ** 2 * tau)
    b = 1 / np.fabs(p - 1)
    c = fwd ** (2 * (1 - p)) / ((1 - p) ** 2 * lamda ** 2 * tau)
    p_cutoff = 0.99e-2

    if abs(p - 1) < p_cutoff:
        return undiscBSOptFwd(isCall, fwd, strike, lamda, tau)
    elif p < 1:
        callPrice = fwd * (1 - ncx2.cdf(a, b + 2, c)) - strike * ncx2.cdf(c, b, a)
    else:
        callPrice = fwd * (1 - ncx2.cdf(c, b, a)) - strike * ncx2.cdf(a, b + 2, c)

    if np.any(ncx2.cdf(a, b + 2, c) > 1) or np.any(ncx2.cdf(c, b, a) > 1):
        raise RuntimeError(f"CDF greater than 1: p={p}. Maybe too close to 0 or 1?")
    elif np.any(callPrice < 0):
        raise RuntimeError(
            f"Negative option price detected: p={p}. Maybe too close to 0 or 1?"
        )

    if isCall:
        return callPrice
    else:
        return callPrice - fwd + strike


def CEVSpot(isCall, spot, strike, tau, lamda, p, r, q):
    """The CEV model is given by John Hull on page 625.
    dS = (r-q)S dt + lamda S^p dW(t)
    where r is risk free rate and q is the dividend yield.
    """
    p_cutoff = 0.99e-2
    if abs(p - 1) < p_cutoff:
        return BSOpt(isCall, spot, strike, lamda, tau, r, q)

    if np.fabs(r - q) < 1e-6:
        v = lamda ** 2 * tau
    else:
        v = (
            lamda ** 2
            / (2 * (r - q) * (p - 1))
            * (np.exp(2 * (r - q) * (p - 1) * tau) - 1)
        )
    a = (strike * np.exp(-(r - q) * tau)) ** (2 * (1 - p)) / ((1 - p) ** 2 * v)
    b = 1 / np.fabs(p - 1)
    c = spot ** (2 * (1 - p)) / ((1 - p) ** 2 * v)

    if p < 1:
        if isCall:
            Price = spot * np.exp(-q * tau) * (
                1 - ncx2.cdf(a, b + 2, c)
            ) - strike * np.exp(-r * tau) * ncx2.cdf(c, b, a)
        else:
            Price = strike * np.exp(-r * tau) * (1 - ncx2.cdf(c, b, a)) - spot * np.exp(
                -q * tau
            ) * ncx2.cdf(a, b + 2, c)
    else:
        if isCall:
            Price = spot * np.exp(-q * tau) * (1 - ncx2.cdf(c, b, a)) - strike * np.exp(
                -r * tau
            ) * ncx2.cdf(a, b + 2, c)
        else:
            Price = strike * np.exp(-r * tau) * (
                1 - ncx2.cdf(a, b + 2, c)
            ) - spot * np.exp(-q * tau) * ncx2.cdf(c, b, a)

    if np.any(ncx2.cdf(a, b + 2, c) > 1) or np.any(ncx2.cdf(c, b, a) > 1):
        raise RuntimeError(f"CDF greater than 1: p={p}. Maybe too close to 0 or 1?")
    elif np.any(Price < 0):
        raise RuntimeError(
            f"Negative option price detected: p={p}. Maybe too close to 0 or 1?"
        )

    return Price


class customFunc:
    """Define the customerized functors for functions used in local vol model, including:
    mu(t,x)     :   the drift function
    sigma(t,x)  :   the diffusion function
    r(t,x)      :   the process for short rates"""

    def __init__(self, category, **kwargs):
        self.category = category
        self.params = kwargs

    def __call__(self, t, x_t):
        if self.category.upper() == "CONSTANT":
            a = self.params.get("a", None)
            if a is not None:
                return a * np.ones_like(x_t)
        elif self.category.upper() == "LINEAR":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            if not None in [a, b]:
                return a + b * x_t
        elif self.category.upper() == "RELU":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            if not None in [a, b]:
                return np.maximum(a + b * x_t, 0)
        elif self.category.upper() == "CEV":
            lamda = self.params.get("lamda", None)
            p = self.params.get("p", None)
            if not None in [lamda, p]:
                return lamda * x_t ** p
        elif self.category.upper() == "EXP":
            c = self.params.get("c", None)
            d = self.params.get("d", None)
            if not None in [c, d]:
                return c * np.exp(d * x_t)
        elif self.category.upper() == "EXP RELU":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            c = self.params.get("c", None)
            if not None in [a, b, c]:
                return np.maximum(a * np.exp(b * x_t) + c, 0)
        elif self.category.upper() == "EXP DIFF":
            # call: x_t*e^{-q*(T-t)} - k*e^{-r*(T-t)}
            # put:  k*e^{-r*(T-t)} - x_t*e^{-q*(T-t)}
            isCall = self.params.get("isCall", None)
            k = self.params.get("k", None)
            r = self.params.get("r", None)
            q = self.params.get("q", None)
            T = self.params.get("T", None)
            if not None in [isCall, k, r, q, T]:
                if isCall:
                    return x_t * np.exp(-q * (T - t)) - k * np.exp(-r * (T - t))
                else:
                    return k * np.exp(-r * (T - t)) - x_t * np.exp(-q * (T - t))
        elif self.category.upper() == "EXP DIFF 2":
            # e^a*e^{-q*(T-t)} - e^k*e^{-r*(T-t)}
            isCall = self.params.get("isCall", None)
            k = self.params.get("k", None)
            r = self.params.get("r", None)
            q = self.params.get("q", None)
            T = self.params.get("T", None)
            if not None in [isCall, k, r, q, T]:
                if isCall:
                    return np.exp(x_t - q * (T - t)) - np.exp(k - r * (T - t))
                else:
                    return np.exp(k - r * (T - t)) - np.exp(x_t - q * (T - t))

        raise ValueError("Input parameters not found!")
