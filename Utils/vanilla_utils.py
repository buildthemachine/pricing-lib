"""
This code implements the different utility functions of vanilla option pricing
Author:     Yufei Shen
Date:       12/21/2020
# pricing-lib

"""

import logging
import math
import numpy as np
import scipy.optimize

from abc import ABCMeta, abstractmethod
from scipy.stats import norm, ncx2
from Utils.other_utils import dictGetAttr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class single_asset_vol_base(metaclass=ABCMeta):
    """Base volatility class for vanilla products
    Supports two functions: price and impliedVol"""

    @abstractmethod
    def __init__(
        self, isCall, x0, strike, tau, ir, dividend_yield, prod_type, **kwargs
    ):
        self.isCall = isCall
        self.x0 = x0  # x0 can be either forward or spot prices
        self.strike = strike
        self.tau = tau
        self.ir = ir
        self.dividend_yield = dividend_yield
        self.prod_type = prod_type

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


class GBM_randomization(single_asset_vol_base):
    """Pricing object for American option on GBM dynamics using Carr randomization"""

    def __init__(
        self,
        isCall,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="AMERICAN VANILLA",
        **kwargs,
    ):
        super().__init__(isCall, spot, strike, tau, ir, 0, prod_type, **kwargs)
        self.vol = dictGetAttr(self.other_params, "vol", None)
        self.richardson_order = dictGetAttr(self.other_params, "richardson_order", 5)
        self.phi = dictGetAttr(self.other_params, "phi", 0)
        self.deta = dividend_yield

    def price(self):
        return carr_randomization_dividend(
            self.isCall,
            self.x0,
            self.strike,
            self.vol,
            self.tau,
            self.ir,
            self.phi,
            self.deta,
            self.richardson_order,
        )


def carr_randomization_dividend(
    isCall, spot, strike, vol, tau, r, phi, deta, richardson_order
):
    """Implements the American put option pricing formula with Richardson extrapolation described in
    'Randomization and the American Put', Carr, 1998"""
    price_by_order = []

    for order in range(1, richardson_order + 1):
        gamma = 0.5 - (r - deta) / vol ** 2
        Delta = tau / order
        R = 1 / (1 + r * Delta)
        D = 1 / (1 + deta * Delta)
        epsilon = math.sqrt(gamma ** 2 + 2 / (R * vol ** 2 * Delta))
        p = (epsilon - gamma) / (2 * epsilon)
        q = 1 - p
        p_hat = (epsilon - gamma + 1) / (2 * epsilon)
        q_hat = 1 - p_hat
        params = [R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon]
        s = spot - phi / r * (1 - R ** order)

        if isCall:
            # if deta == 0 and phi <= r * strike:
            if phi <= (r - deta) * strike:
                price_by_order.append(c_n_S_dividend(s, strike, params, order))
            else:
                exercise_boundary = get_exercise_boundary_call_dividend(
                    strike, params, order
                )
                price_by_order.append(
                    CnS_dividend(spot, s, strike, params, exercise_boundary)
                )
        else:
            if deta == 0 and phi >= r * strike:
                price_by_order.append(p_n_S_dividend(s, strike, params, order))
            else:
                exercise_boundary = get_exercise_boundary_put_dividend(
                    strike, params, order
                )
                price_by_order.append(
                    PnS_dividend(spot, s, strike, params, exercise_boundary)
                )

    price_by_order = np.array(price_by_order)
    return richardson_extrapolation(price_by_order, richardson_order)


def get_exercise_boundary_call_dividend(strike, params, n):
    """Obtain the exercise boundaries for n-Poisson jumps via root finding"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    exercise_boundary = [strike]
    for m in range(1, n + 1):

        def f(x):
            return (
                (strike / x) ** (gamma - epsilon)
                * (q_hat * D * x * deta - q * R * (strike * r - phi))
                * Delta
                - p_n_S_dividend(strike, strike, params, n)
                + B_i_n_Sh_dividend(strike, strike, params, 1, m, 2, exercise_boundary)
            )

        start = strike * (
            (strike * q_hat * D * deta * Delta)
            / (
                p_n_S_dividend(strike, strike, params, n)
                - B_i_n_Sh_dividend(strike, strike, params, 1, m, 2, exercise_boundary)
            )
        ) ** (1 / (gamma - epsilon - 1))
        Sm = scipy.optimize.newton(f, start)
        exercise_boundary.append(Sm)
    return np.array(exercise_boundary)


def alpha_i_n_S_dividend(s, strike, params, i, n, exercise_boundary):
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    res = 0
    for j in range(1, n - i + 2):
        res_mid = 0
        for k in range(j):
            inner_sum = np.sum(
                [
                    math.comb(j - 1 + l, j - 1)
                    * (
                        p_hat ** j
                        * q_hat ** (k + l)
                        * D ** j
                        * exercise_boundary[n - j + 1]
                        * deta
                        - p ** j * q ** (k + l) * R ** j * (strike * r - phi)
                    )
                    * Delta
                    for l in range(j - k)
                ]
            )
            res_mid += (
                (2 * epsilon * math.log(exercise_boundary[n - j + 1] / s)) ** k
                / math.factorial(k)
                * inner_sum
            )
        res += (s / exercise_boundary[n - j + 1]) ** (gamma + epsilon) * res_mid
    return res


def B_i_n_Sh_dividend(s, strike, params, i, n, h, exercise_boundary):
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    res = 0
    for j in range(h, n - i + 2):
        res_mid = 0
        for k in range(j):
            inner_sum = np.sum(
                [
                    math.comb(j - 1 + l, j - 1)
                    * (
                        q_hat ** j
                        * p_hat ** (k + l)
                        * D ** j
                        * exercise_boundary[n - j + 1]
                        * deta
                        - q ** j * p ** (k + l) * R ** j * (strike * r - phi)
                    )
                    * Delta
                    for l in range(j - k)
                ]
            )
            res_mid += (
                (2 * epsilon * math.log(s / exercise_boundary[n - j + 1])) ** k
                / math.factorial(k)
                * inner_sum
            )
        res += (s / exercise_boundary[n - j + 1]) ** (gamma - epsilon) * res_mid
    return res


def CnS_dividend(spot, s, strike, params, exercise_boundary):
    """Return the American call price for n-period jump process"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    index = np.searchsorted(exercise_boundary, s, side="right")
    n = exercise_boundary.size - 1

    if index == 0:
        return c_n_S_dividend(s, strike, params, n) + alpha_i_n_S_dividend(
            s, strike, params, 1, n, exercise_boundary
        )
    elif index == n + 1:
        return spot - strike
    else:
        return (
            -v_i_n_S_dividend(s, strike, R, D, phi, Delta, index, n)
            + alpha_i_n_S_dividend(s, strike, params, index, n, exercise_boundary)
            + B_i_n_Sh_dividend(s, strike, params, index, n, 1, exercise_boundary)
        )


def get_exercise_boundary_put_dividend(strike, params, n):
    """Obtain the exercise boundaries for n-Poisson jumps via Eqn. (49) via root finding"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    exercise_boundary = [strike]
    for m in range(1, n + 1):

        def f(x):
            return (
                (strike / x) ** (gamma + epsilon)
                * (p * R * (strike * r - phi) - p_hat * D * x * deta)
                * Delta
                - c_1_m_K_dividend(strike, params, m)
                + A_i_n_Sh_dividend(strike, strike, params, 1, m, 2, exercise_boundary)
            )

        Sm = scipy.optimize.brentq(f, 1e-8, exercise_boundary[-1])
        exercise_boundary.append(Sm)
    return np.array(exercise_boundary)


def c_1_m_K_dividend(strike, params, m):
    """Eqn. (50) implementation"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    return np.sum(
        [
            math.comb(m - 1 + l, m - 1)
            * (
                strike * D ** m * p_hat ** m * q_hat ** l
                - strike * R ** m * p ** m * q ** l
            )
            for l in range(m)
        ]
    )


def PnS_dividend(spot, s, strike, params, exercise_boundary):
    """Return the American put price for n-period jump process, via Eqn. (46)"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    index = exercise_boundary.size - np.searchsorted(
        exercise_boundary[::-1], s, side="left"
    )
    n = exercise_boundary.size - 1

    if index == 0:
        return p_n_S_dividend(s, strike, params, n) + b_i_n_S_dividend(
            s, strike, params, 1, n, exercise_boundary
        )
    elif index == n + 1:
        return strike - spot
    else:
        return (
            v_i_n_S_dividend(s, strike, R, D, phi, Delta, index, n)
            + b_i_n_S_dividend(s, strike, params, index, n, exercise_boundary)
            + A_i_n_Sh_dividend(s, strike, params, index, n, 1, exercise_boundary)
        )


def A_i_n_Sh_dividend(s, strike, params, i, n, h, exercise_boundary):
    """Eqn. (48.5) implementation"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    res = 0
    for j in range(h, n - i + 2):
        res_mid = 0
        for k in range(j):
            inner_sum = np.sum(
                [
                    math.comb(j - 1 + l, j - 1)
                    * (
                        p ** j * q ** (k + l) * R ** j * (strike * r - phi)
                        - p_hat ** j
                        * q_hat ** (k + l)
                        * D ** j
                        * exercise_boundary[n - j + 1]
                        * deta
                    )
                    * Delta
                    for l in range(j - k)
                ]
            )
            res_mid += (
                (2 * epsilon * math.log(exercise_boundary[n - j + 1] / s)) ** k
                / math.factorial(k)
                * inner_sum
            )
        res += (s / exercise_boundary[n - j + 1]) ** (gamma + epsilon) * res_mid
    return res


def v_i_n_S_dividend(s, strike, R, D, phi, Delta, i, n):
    """Implement the expression for v_i^n(S) in Eqn. (47)"""
    return (
        strike * R ** (n - i + 1)
        - s * D ** (n - i + 1)
        - phi * Delta * R * (R ** (n - i + 1) - R ** n) / (1 - R)
    )


def p_n_S_dividend(s, strike, params, n):
    """Eqn. (43) implementation"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    res = 0
    if s >= strike:
        for k in range(n):
            inner_sum = np.sum(
                [
                    math.comb(n - 1 + l, n - 1)
                    * (
                        strike * R ** n * q ** n * p ** (l + k)
                        - strike * D ** n * q_hat ** n * p_hat ** (l + k)
                    )
                    for l in range(n - k)
                ]
            )
            res += (
                (2 * epsilon * math.log(s / strike)) ** k
                / math.factorial(k)
                * inner_sum
            )
        return res * (s / strike) ** (gamma - epsilon)
    else:
        # return strike * R ** n - s * D ** n + c_n_S_dividend(spot, strike, params, n)
        for k in range(n):
            inner_sum = np.sum(
                [
                    math.comb(n - 1 + l, n - 1)
                    * (
                        strike * D ** n * p_hat ** n * q_hat ** (k + l)
                        - strike * R ** n * p ** n * q ** (k + l)
                    )
                    for l in range(n - k)
                ]
            )
            res += (
                (2 * epsilon * math.log(strike / s)) ** k
                / math.factorial(k)
                * inner_sum
            )
        return res * (s / strike) ** (gamma + epsilon) + strike * R ** n - s * D ** n


def c_n_S_dividend(s, strike, params, n):
    """Implement Equation (44) via put call parity"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    return s * D ** n - strike * R ** n + p_n_S_dividend(s, strike, params, n)


def b_i_n_S_dividend(s, strike, params, i, n, exercise_boundary):
    """Eqn. (48) implementation"""
    R, r, D, phi, deta, Delta, p, q, p_hat, q_hat, gamma, epsilon = params
    res = 0
    for j in range(1, n - i + 2):
        res_mid = 0
        for k in range(j):
            inner_sum = np.sum(
                [
                    math.comb(j - 1 + l, j - 1)
                    * (
                        q ** j * p ** (k + l) * R ** j * (strike * r - phi)
                        - q_hat ** j
                        * p_hat ** (k + l)
                        * D ** j
                        * exercise_boundary[n - j + 1]
                        * deta
                    )
                    * Delta
                    for l in range(j - k)
                ]
            )
            res_mid += (
                (2 * epsilon * math.log(s / exercise_boundary[n - j + 1])) ** k
                / math.factorial(k)
                * inner_sum
            )
        res += (s / exercise_boundary[n - j + 1]) ** (gamma - epsilon) * res_mid
    return res


def richardson_extrapolation(price_by_order, richardson_order):
    """Implement Richardson extrapolation as in Eqn. (35) of the Carr Randomization paper"""
    try:
        assert len(price_by_order) == richardson_order
    except AssertionError as e:
        logger.error("The length of the prices should equal the richardson order!")
    prices = [
        richardson_summand(n, richardson_order, price_by_order[n - 1])
        for n in range(1, richardson_order + 1)
    ]
    return np.sum(prices)


def richardson_summand(n, N, P_Tn):
    return (-1) ** (N - n) * n ** N / (math.factorial(n) * math.factorial(N - n)) * P_Tn


class Bachlier_obj(single_asset_vol_base):
    """Pricing object for Bachelier model"""

    def __init__(
        self,
        isCall,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="EUROPEAN VANILLA",
        **kwargs,
    ):
        super().__init__(
            isCall, spot, strike, tau, ir, dividend_yield, prod_type, **kwargs
        )
        self.vol = dictGetAttr(self.other_params, "vol", None)

    def price(self):
        return BachlierSpot(
            self.isCall, self.x0, self.strike, self.vol, self.ir, self.tau
        )


class CEV_obj(single_asset_vol_base):
    """Pricing object for CEV model"""

    def __init__(
        self,
        isCall,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="EUROPEAN VANILLA",
        **kwargs,
    ):
        super().__init__(
            isCall, spot, strike, tau, ir, dividend_yield, prod_type, **kwargs
        )
        self.lamda = dictGetAttr(self.other_params, "lamda", None)
        self.p = dictGetAttr(self.other_params, "p", None)

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


class GBM_obj(single_asset_vol_base):
    """Pricing object for Black-Scholes equation"""

    def __init__(
        self,
        isCall,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="EUROPEAN VANILLA",
        **kwargs,
    ):
        super().__init__(
            isCall,
            spot,
            strike,
            tau,
            ir,
            dividend_yield,
            prod_type,
            **kwargs,
        )
        self.vol = dictGetAttr(self.other_params, "vol", None)

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


class GBM_digital_obj(single_asset_vol_base):
    """Pricing object for European digital options via Black-Scholes"""

    def __init__(
        self,
        isCall,
        isCashOrNothing,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="EUROPEAN DIGITAL",
        **kwargs,
    ):
        super().__init__(
            isCall,
            spot,
            strike,
            tau,
            ir,
            dividend_yield,
            prod_type,
            **kwargs,
        )
        self.isCashOrNothing = isCashOrNothing
        self.vol = dictGetAttr(self.other_params, "vol", None)

    def price(self):
        return BS_digital_Opt(
            self.isCall,
            self.isCashOrNothing,
            self.x0,
            self.strike,
            self.vol,
            self.tau,
            self.ir,
            self.dividend_yield,
        )


class CEV_digital_obj(single_asset_vol_base):
    """Pricing object for European digital options with CEV dynamics"""

    def __init__(
        self,
        isCall,
        isCashOrNothing,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="EUROPEAN DIGITAL",
        **kwargs,
    ):
        super().__init__(
            isCall,
            spot,
            strike,
            tau,
            ir,
            dividend_yield,
            prod_type,
            **kwargs,
        )
        self.isCashOrNothing = isCashOrNothing
        self.cev_p = dictGetAttr(self.other_params, "p", None)
        self.cev_lamda = dictGetAttr(self.other_params, "lamda", None)

    def impliedVol(self):
        euro_price = CEVSpot(
            self.isCall,
            self.x0,
            self.strike,
            self.tau,
            self.cev_lamda,
            self.cev_p,
            self.ir,
            self.dividend_yield,
        )
        self.bsvol = BSImpVol(
            self.isCall,
            self.x0,
            self.strike,
            self.tau,
            euro_price,
            self.ir,
            self.dividend_yield,
        )
        return self.bsvol

    def price(self):
        return BS_digital_Opt(
            self.isCall,
            self.isCashOrNothing,
            self.x0,
            self.strike,
            self.impliedVol(),
            self.tau,
            self.ir,
            self.dividend_yield,
        )


class SABR_digital_obj(single_asset_vol_base):
    """Pricing object for European digital options with SABR
    We first obtain the Black volatility from sabr dynamics, then plug that
    volatility into digital option pricing formula"""

    def __init__(
        self,
        isCall,
        isCashOrNothing,
        spot,
        strike,
        tau,
        ir,
        dividend_yield,
        prod_type="EUROPEAN DIGITAL",
        **kwargs,
    ):
        super().__init__(
            isCall,
            spot,
            strike,
            tau,
            ir,
            dividend_yield,
            prod_type,
            **kwargs,
        )
        self.isCashOrNothing = isCashOrNothing
        self.sabr_params = dictGetAttr(self.other_params, "SABR_params", None)

    def impliedVol(self):
        sabr_obj = sabr_black_vol(
            self.isCall, self.x0, self.strike, self.tau, self.sabr_params
        )
        self.black_vol = sabr_black_vol.impliedVol()
        return self.black_vol

    def price(self):
        return BS_digital_Opt(
            self.isCall,
            self.isCashOrNothing,
            self.x0,
            self.strike,
            self.impliedVol(),
            self.tau,
            self.ir,
            self.dividend_yield,
        )


class sabr_black_vol(single_asset_vol_base):
    """TODO: Writing unit test files for sabr
    SABR model does not depend on interest rate, therefore set r=q=0"""

    def __init__(
        self,
        isCall,
        x0,
        strike,
        tau,
        ir,
        dividend_yield,
        sabr_params,
        prod_type="EUROPEAN DIGITAL",
        eps=1e-6,
    ):
        """This methods provides an interface for constructing a vol smile.
        The strike k provided can be a vector."""
        super().__init__(isCall, x0, strike, tau, ir, dividend_yield, prod_type)
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


def BS_digital_Opt(isCall, isCashOrNothing, spot, strike, vol, tau, r, q):
    """
    r: risk free interest rate
    q: convenience yield (-) or dividend payment (+)
    """
    if isCashOrNothing:
        if isCall:
            return np.exp(-r * tau) * norm.cdf(d2(spot, strike, vol, tau, r, q))
        else:
            return np.exp(-r * tau) * norm.cdf(-d2(spot, strike, vol, tau, r, q))
    else:
        if isCall:
            return spot * np.exp(-q * tau) * norm.cdf(d1(spot, strike, vol, tau, r, q))
        else:
            return spot * np.exp(-q * tau) * norm.cdf(-d1(spot, strike, vol, tau, r, q))


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
