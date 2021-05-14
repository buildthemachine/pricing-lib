"""
This code implements the different utility functions of local volatility.
Author:     Yufei Shen
Date:       12/30/2020
# pricing-lib

Contains pricing functionalities of financial derivatives. Several pricing engines are supported including:
- Analytical solutions
- PDE method (backward induction)
- Monte Carlo
"""

import logging
import numpy as np

from numba import njit, double

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

frequency_counts_dict = {
    "HOURLY": 262 * 8,
    "DAILY": 262,
    "WEEKLY": 52,
    "MONTHLY": 12,
    "QUARTERLY": 4,
    "ANNUALLY": 1,
}


@njit(double(double))
def cnorma(x):
    """A double precision algorithm for univariate cumulative normal.
    Refer to 'Better approximations to cumulative normal functions' by Graeme West"""
    XAbs = abs(x)
    if XAbs > 37.0:
        Cumnorm = 0
    else:
        Exponential = np.exp(-XAbs * XAbs / 2)
        if XAbs < 7.07106781186547:
            build = 3.52624965998911e-02 * XAbs + 0.700383064443688
            build = build * XAbs + 6.37396220353165
            build = build * XAbs + 33.912866078383
            build = build * XAbs + 112.079291497871
            build = build * XAbs + 221.213596169931
            build = build * XAbs + 220.206867912376
            Cumnorm = Exponential * build
            build = 8.83883476483184e-02 * XAbs + 1.75566716318264
            build = build * XAbs + 16.064177579207
            build = build * XAbs + 86.7807322029461
            build = build * XAbs + 296.564248779674
            build = build * XAbs + 637.333633378831
            build = build * XAbs + 793.826512519948
            build = build * XAbs + 440.413735824752
            Cumnorm = Cumnorm / build
        else:
            build = XAbs + 0.65
            build = XAbs + 4 / build
            build = XAbs + 3 / build
            build = XAbs + 2 / build
            build = XAbs + 1 / build
            Cumnorm = Exponential / build / 2.506628274631

    if x > 0:
        Cumnorm = 1 - Cumnorm

    return Cumnorm


def dictGetAttr(dic, key, val):
    """This method obtains class attributes from the input key-word dictionary by key lookup.
    If key nonexist, prints a warning message and use default value"""
    key = key.upper()
    if key in dic:
        return dic.pop(key)
    else:
        logger.warning(f"The {key} attribute is not provided. Use default value {val}.")
        return val


class customFunc:
    """Define the customerized functors for functions used in local vol model, including:
    mu(t,x)     :   the drift function
    sigma(t,x)  :   the diffusion function
    r(t,x)      :   the process for short rates"""

    def __init__(self, category, **kwargs):
        self.category = category.upper()
        self.params = kwargs

    def __call__(self, t, x_t):
        if self.category == "CONSTANT":
            a = self.params.get("a", None)
            if a is not None:
                return a * np.ones_like(x_t)
        elif self.category == "LINEAR":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            if not None in [a, b]:
                return a + b * x_t
        elif self.category == "STEP":
            values = self.params.get("values", None)
            intervals = self.params.get("intervals", None)
            side = self.params.get("side", "right")
            assert (
                len(values) == len(intervals) + 1
            ), "In STEP func: # values does not match # intervals!"
            assert np.all(
                np.diff(intervals) < 0
            ), "In STEP func: interval boundaries should be monotonic!"
            idx = np.searchsorted(intervals, x_t, side=side)
            return np.take(values, idx)
        elif self.category == "RELU":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            if not None in [a, b]:
                return np.maximum(a + b * x_t, 0)
        elif self.category == "CEV":
            lamda = self.params.get("lamda", None)
            p = self.params.get("p", None)
            if not None in [lamda, p]:
                return lamda * x_t ** p
        elif self.category == "EXP":
            c = self.params.get("c", None)
            d = self.params.get("d", None)
            if not None in [c, d]:
                return c * np.exp(d * x_t)
        elif self.category == "EXP RELU":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            c = self.params.get("c", None)
            if not None in [a, b, c]:
                return np.maximum(a * np.exp(b * x_t) + c, 0)
        elif self.category == "EXP TIME":
            a = self.params.get("a", None)
            b = self.params.get("b", None)
            c = self.params.get("c", None)
            if not None in [a, b, c]:
                return a * np.exp(b + c * t)
        elif self.category == "EXP DIFF":
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
        elif self.category == "EXP DIFF 2":
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
        elif self.category == "BI-PIECEWISE":
            x0 = self.params.get("middle_point", None)
            left_func = self.params.get("left_functor", None)
            right_func = self.params.get("right_functor", None)
            side = self.params.get("side", "right").upper()
            if side not in ["LEFT", "RIGHT"]:
                raise ValueError("Side has unexpected value!")
            if side == "RIGHT":
                mask = x_t <= x0
            else:
                mask = x_t < x0
            res_left = left_func(t, x_t[mask])
            res_right = right_func(t, x_t[~mask])
            return np.concatenate((res_left, res_right))

        raise ValueError(
            f"Input category {self.category} not found, or some required parameters are missing!"
        )


if __name__ == "__main__":
    unittest.main()
