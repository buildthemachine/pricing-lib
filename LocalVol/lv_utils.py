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
import scipy.optimize
import scipy.interpolate
import scipy.sparse
from scipy.stats import norm, ncx2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def dictGetAttr(dic, key, val):
    """This method obtains class attributes from the input key-word dictionary by key lookup.
    If key nonexist, prints a warning message and use default value"""
    key = key.upper()
    if key in dic:
        return dic.pop(key)
    else:
        logger.warning(f"The {key} attribute is not provide. Use default value {val}.")
        return val


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
    r            : risk free interst rate
    f_up(t,x)    : functor for upper boundary condition for x=M_high
    f_dn(t,x)    : functor for lower boundary condition for x=M_low
    g(x)         : terminal condition at time t=T
    """

    def __init__(self, tau, underlying, **kwargs):
        self.tau = tau
        self.x0 = underlying  # Time 0 value of the underlying
        kwargs = {
            k.upper(): v for k, v in kwargs.items()
        }  # Convert all keys to upper case.
        self._Mhigh = dictGetAttr(kwargs, "Mhigh", self.x0 * 3)
        self._Mlow = dictGetAttr(kwargs, "Mlow", -self.x0)
        self._theta = dictGetAttr(kwargs, "theta", 0.5)
        self._equiGrid = dictGetAttr(kwargs, "EquidistantGrid", True)
        self._interpFlag = dictGetAttr(kwargs, "interp", "linear")
        if self._equiGrid:
            self._m = dictGetAttr(kwargs, "m", 10)  # x grid discritization
            self._n = dictGetAttr(kwargs, "n", 10)  # time grid discritization
            self._tGrid, self._xGrid = self._genGrid()
        else:
            self._tGrid = dictGetAttr(kwargs, "time grid", None)
            self._xGrid = dictGetAttr(kwargs, "x grid", None)
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

    def _genGrid(self):
        """Generate equidistant t and x grids using boundary values as well as # of discretizations m/n.
        This discretization is consistent with Andersen book page 45."""
        tGrid = np.linspace(0, self.tau, self._n + 1)
        xGrid = np.linspace(self._Mlow, self._Mhigh, self._m + 2)
        return tGrid, xGrid
