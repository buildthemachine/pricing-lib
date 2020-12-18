"""
This code implements the different utility functions of local volatility.
Author:     Yufei Shen
Date:       11/17/2020
# pricing-lib

Contains pricing functionalities of financial derivatives. Several pricing engines are supported including:
- Analytical solutions
- PDE method (backward induction)
- Monte Carlo
"""

import numpy as np
import scipy.optimize, scipy.interpolate, scipy.sparse
from scipy.stats import norm, ncx2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    dS(t)=\lambda dW(t)
    where \lambda is the volatilty
    """
    d = d_Bache(fwd, strike, vol, tau)
    callPrice = (fwd - strike) * norm.cdf(d) + vol * np.sqrt(tau) * norm.pdf(d)
    if isCall:
        return callPrice
    else:
        return callPrice - fwd + strike  # From put-call parity


def BlackImpVol(isCall, discount, fwd, strike, tau, price):
    """Back out the implied volatility from Black's model, through a root finding approach
    For undiscounted pricing as input, set discount = 1."""

    def loss(vol):
        return undiscBSOptFwd(isCall, fwd, strike, vol, tau) - price / discount

    volMin = 1.0e-6
    volMax = 10
    Vol = scipy.optimize.brenth(loss, volMin, volMax)
    return Vol


def undiscCEVFwd(isCall, fwd, strike, lamda, tau, p):
    """The CEV model is given by equation 7.9 in Andersen. In particular:
    dS(t)=\lambda S(t)^p dW(t)
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


def CEVSpot(isCall, spot, strike, lamda, tau, p, r, q):
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
            # a*e^{-q*(T-t)} - b*e^{-r*(T-t)}
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


class Mesh:
    """
    Class to implement backward induction in a PDE solver.
    tau      :   time-to-maturity
    theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
    M_high   :   upper bound in underlying prices
    M_low    :   lower bound in underlying prices
    m, n     :   discretization in underlying/time domain
    mu(t,x)  :   functors representing the time-dependeint drift coefficients
    sigma(t,x)   : functors representing the time-dependeint diffusion coefficients
    f_up(t,x)    : functor for upper boundary condition for x=M_high
    f_dn(t,x)    : functor for lower boundary condition for x=M_low
    """

    def __init__(self, tau, underlying, **kwargs):
        self.tau = tau
        self.x0 = underlying  # Time 0 value of the underlying
        kwargs = {
            k.upper(): v for k, v in kwargs.items()
        }  # Convert all keys to upper case.
        self._Mhigh = self._getAttr(kwargs, "Mhigh", self.x0 * 3)
        self._Mlow = self._getAttr(kwargs, "Mlow", -self.x0)
        self._theta = self._getAttr(kwargs, "theta", 0.5)
        self._equiGrid = self._getAttr(kwargs, "EquidistantGrid", True)
        self._interpFlag = self._getAttr(kwargs, "interp", "linear")
        if self._equiGrid:
            self._m = self._getAttr(kwargs, "m", 10)  # x grid discritization
            self._n = self._getAttr(kwargs, "n", 10)  # time grid discritization
            self._tGrid, self._xGrid = self._genGrid()
        else:
            self._tGrid = _getAttr(kwargs, "time grid", None)
            self._xGrid = _getAttr(kwargs, "x grid", None)
            assert (
                self._tGrid and self._xGrid
            )  # Check the grids have been successfully created
            self._m = len(self._xGrid) - 2  # See Andersen page 45 for '-2'
            self._n = len(self._tGrid) - 1  # See Andersen page 45 for '-1'

        assert self._tGrid[0] == 0 and self._tGrid[-1] == self.tau
        assert self._xGrid[0] == self._Mlow and self._xGrid[-1] == self._Mhigh
        func_dict = {
            "mu": "drift",
            "sigma": "diffusion",
            "r": "short rate",
            "f_up": "upper bound",  # x->_Mhigh bound
            "f_dn": "lower bound",  # x->_Mlow bound
            "g": "terminal condition",
        }
        for k, v in func_dict.items():
            if k.upper() in kwargs:
                self.__dict__["_" + k] = kwargs.pop(k.upper())
            else:
                logger.error(f"The {v} functor {k}(t,x) has not been provided!")
                raise KeyError("Please fix the input and come back!")
        # time 0 vector V(0,x). This is solved during the init stage so that in __call__ we only need to perform interpolation.
        self._V0 = self._backwardInduction()

    def _backwardInduction(self):
        """This is the backward induction (in time) method used to solve the PDE:
        partial V/partial t + AV=0"""
        V0 = np.zeros((self._n + 1, self._m))
        V0[self._n] = self._g(self.tau, self._xGrid[1:-1])
        omega = np.zeros((self._n + 1, self._m))  # omega controls the boundary

        for i in range(self._n)[::-1]:  # Going backward in time
            t_theta = (1 - self._theta) * self._tGrid[
                i + 1
            ] + self._theta * self._tGrid[i]
            delta_t = self._tGrid[i + 1] - self._tGrid[i]
            # All below vectors are m-dimensional
            delta_x_plus = np.diff(self._xGrid)[1:]
            delta_x_minus = np.diff(self._xGrid)[:-1]
            mu = self._mu(t_theta, self._xGrid[1:-1])
            sigma = self._sigma(t_theta, self._xGrid[1:-1])
            ir = self._r(t_theta, self._xGrid[1:-1])
            c_vec = (
                (delta_x_plus - delta_x_minus) / (delta_x_plus * delta_x_minus)
                - 1 / (delta_x_plus * delta_x_minus) * sigma ** 2
                - ir
            )
            u_vec = (
                delta_x_minus / (delta_x_plus * (delta_x_plus + delta_x_minus)) * mu
                + 1 / (delta_x_plus * (delta_x_plus + delta_x_minus)) * sigma ** 2
            )
            l_vec = (
                -delta_x_plus / (delta_x_minus * (delta_x_plus + delta_x_minus)) * mu
                + 1 / (delta_x_minus * (delta_x_plus + delta_x_minus)) * sigma ** 2
            )
            # Calculating omega[n] as in Eqn. (2.10):
            omega = np.zeros(self._m)
            omega[0] = l_vec[0] * self._f_dn(t_theta, self._Mlow)
            omega[-1] = u_vec[-1] * self._f_up(t_theta, self._Mhigh)
            A_mat = scipy.sparse.diags(
                [c_vec, u_vec[:-1], l_vec[1:]], [0, 1, -1]
            ).toarray()

            # Write 2.18 as: T*V(t_i)=S
            T = np.eye(self._m) - self._theta * delta_t * A_mat
            S = (
                np.matmul(
                    np.eye(self._m) + (1 - self._theta) * delta_t * A_mat, V0[i + 1]
                )
                + delta_t * omega
            )
            V0[i] = np.matmul(np.linalg.inv(T), S)

        return V0[0]

    def __call__(self, underlying):
        """Define function call method"""
        if self._interpFlag.upper() == "LINEAR":
            return np.interp(underlying, self._xGrid[1:-1], self._V0)
        elif self._interpFlag.upper() == "CUBIC SPLINE":
            cs = scipy.interpolate.CubicSpline(self._xGrid[1:-1], self._V0)
            return cs(underlying)

    def _getAttr(self, dic, key, val):
        """This method obtains class attributes from the input key-word dictionary by key lookup.
        If key nonexist, prints a warning message and use default value"""
        key = key.upper()
        if key in dic:
            return dic.pop(key)
        else:
            logger.warning(
                f"The {key} attribute is not provide. Use default value {val}."
            )
            return val

    def _genGrid(self):
        """Generate equidistant t and x grids using boundary values as well as # of discretizations m/n.
        This discretization is consistent with Andersen book page 45."""
        tGrid = np.linspace(0, self.tau, self._n + 1)
        xGrid = np.linspace(self._Mlow, self._Mhigh, self._m + 2)
        return tGrid, xGrid
