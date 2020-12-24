"""
This code implements the different utility functions of local volatility.
Author:     Yufei Shen
Date:       11/17/2020
"""

from abc import ABCMeta, abstractmethod
from Utils.vanilla_utils import vanilla_vol_base, customFunc
from .lv_utils import Mesh
from LocalVol.lv_utils import dictGetAttr
import numpy as np


class vanilla_localvol_base(vanilla_vol_base):
    """This is the local volatility base class"""

    def __init__(self, isCall, x0, strike, tau, **kwargs):
        super().__init__(isCall, x0, strike, tau)
        kwargs = {
            k.upper(): v for k, v in kwargs.items()
        }  # Convert all keys to upper case.
        self.other_params = kwargs
        self.isLog = dictGetAttr(self.other_params, "isLog", False)

    def price(self):
        if not self.isLog:
            return self.mesh(self.x0)
        else:
            return self.mesh(np.log(self.x0))


class vanilla_localvol_BS(vanilla_localvol_base):
    """Local vol price for Black-Scholes"""

    def __init__(self, isCall, x0, strike, tau, ir, dividend_yield, **kwargs):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """
        super().__init__(isCall, x0, strike, tau, **kwargs)
        self.ir = ir
        self.dividend_yield = dividend_yield
        self.bsvol = dictGetAttr(self.other_params, "vol", None)
        if not self.isLog:
            r = customFunc("constant", a=self.ir)
            mu = customFunc("linear", a=0, b=self.ir - self.dividend_yield)
            # b is BS volatility
            sigma = customFunc("linear", a=0, b=self.bsvol)
            if self.isCall:
                f_up = customFunc(
                    "Exp Diff",
                    isCall=self.isCall,
                    k=self.strike,
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                f_dn = customFunc("constant", a=0)
                g = customFunc("RELU", a=-self.strike, b=1)
            else:
                f_up = customFunc("constant", a=0)
                f_dn = customFunc(
                    "Exp Diff",
                    isCall=self.isCall,
                    k=self.strike,
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                g = customFunc("RELU", a=self.strike, b=-1)
        else:
            r = customFunc("constant", a=self.ir)
            mu = customFunc(
                "constant", a=self.ir - self.dividend_yield - 0.5 * self.bsvol ** 2
            )
            sigma = customFunc("constant", a=self.bsvol)
            if self.isCall:
                f_up = customFunc(
                    "Exp Diff 2",
                    isCall=self.isCall,
                    k=np.log(self.strike),
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                f_dn = customFunc("constant", a=0)
                g = customFunc("Exp RELU", a=1, b=1, c=-self.strike)
            else:
                f_up = customFunc("constant", a=0)
                f_dn = customFunc(
                    "Exp Diff 2",
                    isCall=self.isCall,
                    k=np.log(self.strike),
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                g = customFunc("Exp RELU", a=-1, b=1, c=self.strike)

        self.mesh = Mesh(
            self.tau,
            self.x0,
            r=r,
            mu=mu,
            sigma=sigma,
            f_up=f_up,
            f_dn=f_dn,
            g=g,
            **self.other_params
        )


class vanilla_localvol_CEV(vanilla_localvol_base):
    """Local vol price for CEV"""

    def __init__(self, isCall, x0, strike, tau, ir, dividend_yield, **kwargs):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """
        super().__init__(isCall, x0, strike, tau, **kwargs)
        self.ir = ir
        self.dividend_yield = dividend_yield
        self.cev_lamda = dictGetAttr(self.other_params, "lamda", None)
        self.cev_p = dictGetAttr(self.other_params, "p", None)
        if self.isLog:
            # TODO: implement CEV for log price
            raise NotImplementedError(
                "log version of CEV pricer has yet to be impleted!"
            )

        r = customFunc("constant", a=self.ir)
        mu = customFunc("linear", a=0, b=self.ir - self.dividend_yield)
        sigma = customFunc("cev", lamda=self.cev_lamda, p=self.cev_p)
        if self.isCall:
            f_up = customFunc(
                "Exp Diff",
                isCall=self.isCall,
                k=self.strike,
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            )
            f_dn = customFunc("constant", a=0)
            g = customFunc("RELU", a=-self.strike, b=1)
        else:
            f_up = customFunc("constant", a=0)
            f_dn = customFunc(
                "Exp Diff",
                isCall=self.isCall,
                k=self.strike,
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            )
            g = customFunc("RELU", a=self.strike, b=-1)

        self.mesh = Mesh(
            self.tau,
            self.x0,
            r=r,
            mu=mu,
            sigma=sigma,
            f_up=f_up,
            f_dn=f_dn,
            g=g,
            **self.other_params
        )
