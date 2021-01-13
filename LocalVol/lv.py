"""
This code implements the different utility functions of local volatility.
Author:     Yufei Shen
Date:       11/17/2020
"""

from abc import ABCMeta, abstractmethod
from Utils.vanilla_utils import single_asset_vol_base
from Utils.other_utils import customFunc
from .lv_utils import Mesh
from LocalVol.lv_utils import cevMixin, gbmMixin
from Utils.vanilla_utils import GBM_obj, CEV_obj
from Utils.other_utils import dictGetAttr
import numpy as np


class vanilla_localvol_base(single_asset_vol_base):
    """This is the local volatility base class for vanilla options"""

    def __init__(self, isCall, x0, strike, tau, ir, dividend_yield, **kwargs):
        super().__init__(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)
        self.isLog = dictGetAttr(self.other_params, "isLog", False)
        if not self.isLog:
            if self.isCall:
                self.f_up = customFunc(
                    "Exp Diff",
                    isCall=self.isCall,
                    k=self.strike,
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                self.f_dn = customFunc("constant", a=0)
                self.g = customFunc("RELU", a=-self.strike, b=1)
            else:
                self.f_up = customFunc("constant", a=0)
                self.f_dn = customFunc(
                    "Exp Diff",
                    isCall=self.isCall,
                    k=self.strike,
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                self.g = customFunc("RELU", a=self.strike, b=-1)
        else:
            if self.isCall:
                self.f_up = customFunc(
                    "Exp Diff 2",
                    isCall=self.isCall,
                    k=np.log(self.strike),
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                self.f_dn = customFunc("constant", a=0)
                self.g = customFunc("Exp RELU", a=1, b=1, c=-self.strike)
            else:
                self.f_up = customFunc("constant", a=0)
                self.f_dn = customFunc(
                    "Exp Diff 2",
                    isCall=self.isCall,
                    k=np.log(self.strike),
                    r=self.ir,
                    q=self.dividend_yield,
                    T=self.tau,
                )
                self.g = customFunc("Exp RELU", a=-1, b=1, c=self.strike)

    def price(self):
        if not self.isLog:
            return self.mesh(self.x0)
        else:
            return self.mesh(np.log(self.x0))


class vanilla_localvol_GBM(gbmMixin, vanilla_localvol_base):
    """Local vol price for Black-Scholes
    Mixin class must come as the first parameter"""

    def __init__(self, isCall, x0, strike, tau, ir, dividend_yield, **kwargs):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """
        super().__init__(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)


class vanilla_localvol_CEV(cevMixin, vanilla_localvol_base):
    """Local vol price for CEV
    Mixin class must come as the first parameter"""

    def __init__(self, isCall, x0, strike, tau, ir, dividend_yield, **kwargs):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """
        super().__init__(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)


class single_barrier_localvol_base(single_asset_vol_base):
    """This is the local volatility base class for barrier options"""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        x0,
        strike,
        tau,
        barrier,
        ir,
        dividend_yield,
        **kwargs
    ):
        super().__init__(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)
        self.isContinuous = isContinuous
        self.flavor = flavor
        self.barrier = barrier
        self.isLog = dictGetAttr(self.other_params, "isLog", False)
        if self.flavor not in ["DOWN-AND-OUT", "UP-AND-OUT"]:
            raise NotImplementedError(
                "Not supported! For knock-in options? Make sure you are using the Facade class!"
            )
        self.other_params["HASRESETS"] = not self.isContinuous
        if not self.isLog:
            if self.isContinuous:
                if self.flavor == "DOWN-AND-OUT":
                    if self.isCall:
                        self.f_up = customFunc(
                            "Exp Diff",
                            isCall=self.isCall,
                            k=self.strike,
                            r=self.ir,
                            q=self.dividend_yield,
                            T=self.tau,
                        )
                        self.f_dn = customFunc("constant", a=0)
                        g_left = customFunc("constant", a=0)
                        g_right = customFunc("RELU", a=-self.strike, b=1)
                        self.g = customFunc(
                            "BI-PIECEWISE",
                            middle_point=self.barrier,
                            left_functor=g_left,
                            right_functor=g_right,
                            side="Right",
                        )
                    else:
                        self.f_up = customFunc("constant", a=0)
                        self.f_dn = customFunc("constant", a=0)
                        g_left = customFunc("constant", a=0)
                        g_right = customFunc("RELU", a=self.strike, b=-1)
                        self.g = customFunc(
                            "BI-PIECEWISE",
                            middle_point=self.barrier,
                            left_functor=g_left,
                            right_functor=g_right,
                            side="Right",
                        )
                elif self.flavor == "UP-AND-OUT":
                    if self.isCall:
                        self.f_up = customFunc("constant", a=0)
                        self.f_dn = customFunc("constant", a=0)
                        g_left = customFunc("RELU", a=-self.strike, b=1)
                        g_right = customFunc("constant", a=0)
                        self.g = customFunc(
                            "BI-PIECEWISE",
                            middle_point=self.barrier,
                            left_functor=g_left,
                            right_functor=g_right,
                            side="Right",
                        )
                    else:
                        self.f_up = customFunc("constant", a=0)
                        self.f_dn = customFunc(
                            "Exp Diff",
                            isCall=self.isCall,
                            k=self.strike,
                            r=self.ir,
                            q=self.dividend_yield,
                            T=self.tau,
                        )
                        g_left = customFunc("RELU", a=self.strike, b=-1)
                        g_right = customFunc("constant", a=0)
                        self.g = customFunc(
                            "BI-PIECEWISE",
                            middle_point=self.barrier,
                            left_functor=g_left,
                            right_functor=g_right,
                            side="Right",
                        )
            else:
                # If discrete, the boundary and terminal conditions are the same as vanilla options.
                # The reset events are imposed in the backward propaation algorithm.
                self.other_params["BARRIER"] = self.barrier
                self.other_params["FLAVOR"] = self.flavor
                if self.isCall:
                    self.f_up = customFunc(
                        "Exp Diff",
                        isCall=self.isCall,
                        k=self.strike,
                        r=self.ir,
                        q=self.dividend_yield,
                        T=self.tau,
                    )
                    self.f_dn = customFunc("constant", a=0)
                    self.g = customFunc("RELU", a=-self.strike, b=1)
                else:
                    self.f_up = customFunc("constant", a=0)
                    self.f_dn = customFunc(
                        "Exp Diff",
                        isCall=self.isCall,
                        k=self.strike,
                        r=self.ir,
                        q=self.dividend_yield,
                        T=self.tau,
                    )
                    self.g = customFunc("RELU", a=self.strike, b=-1)

    def price(self):
        if not self.isLog:
            return self.mesh(self.x0)
        else:
            return self.mesh(np.log(self.x0))


class single_barrier_localvol_CEV(cevMixin, single_barrier_localvol_base):
    """Local vol price for Constant Elasticity Volatility
    Mixin class must come as the first parameter
    Only knock-out options are calculated in this class. For knock-in options, their pricing
    relies on in-out parity. The user interface should be the Facade class rather than this one."""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        x0,
        strike,
        tau,
        barrier,
        ir,
        dividend_yield,
        **kwargs
    ):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """

        super().__init__(
            isCall=isCall,
            isContinuous=isContinuous,
            flavor=flavor.upper(),
            x0=x0,
            strike=strike,
            tau=tau,
            barrier=barrier,
            ir=ir,
            dividend_yield=dividend_yield,
            **kwargs,
        )


class single_barrier_localvol_CEV_Facade:
    """This is the user interface to single-barrier options"""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        x0,
        strike,
        tau,
        barrier,
        ir,
        dividend_yield,
        **kwargs
    ):
        self.flavor = flavor.upper()
        if flavor in ["DOWN-AND-OUT", "UP-AND-OUT"]:
            self.barrierObj = single_barrier_localvol_CEV(
                isCall,
                isContinuous,
                flavor,
                x0,
                strike,
                tau,
                barrier,
                ir,
                dividend_yield,
                **kwargs,
            )
        elif flavor == "DOWN-AND-IN":
            self.barrierObj = single_barrier_localvol_CEV(
                isCall,
                isContinuous,
                "DOWN-AND-OUT",
                x0,
                strike,
                tau,
                barrier,
                ir,
                dividend_yield,
                **kwargs,
            )
            self.CEVObj = CEV_obj(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)
        elif flavor == "UP-AND-IN":
            self.barrierObj = single_barrier_localvol_CEV(
                isCall,
                isContinuous,
                "UP-AND-OUT",
                x0,
                strike,
                tau,
                barrier,
                ir,
                dividend_yield,
                **kwargs,
            )
            self.CEVObj = CEV_obj(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)

    def price(self):
        if self.flavor in ["DOWN-AND-OUT", "UP-AND-OUT"]:
            return self.barrierObj.price()
        else:
            value = self.CEVObj.price() - self.barrierObj.price()
            return value


class single_barrier_localvol_GBM(gbmMixin, single_barrier_localvol_base):
    """Local vol price for Geometric Brownian motion
    Mixin class must come as the first parameter
    Only knock-out options are calculated in this class. For knock-in options, their pricing
    relies on in-out parity. The user interface should be the Facade class rather than this one."""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        x0,
        strike,
        tau,
        barrier,
        ir,
        dividend_yield,
        **kwargs
    ):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """

        super().__init__(
            isCall=isCall,
            isContinuous=isContinuous,
            flavor=flavor.upper(),
            x0=x0,
            strike=strike,
            tau=tau,
            barrier=barrier,
            ir=ir,
            dividend_yield=dividend_yield,
            **kwargs,
        )


class single_barrier_localvol_GBM_Facade:
    """This is the user interface to single-barrier options"""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        x0,
        strike,
        tau,
        barrier,
        ir,
        dividend_yield,
        **kwargs
    ):
        self.flavor = flavor.upper()
        if flavor in ["DOWN-AND-OUT", "UP-AND-OUT"]:
            self.barrierObj = single_barrier_localvol_GBM(
                isCall,
                isContinuous,
                flavor,
                x0,
                strike,
                tau,
                barrier,
                ir,
                dividend_yield,
                **kwargs,
            )
        elif flavor == "DOWN-AND-IN":
            self.barrierObj = single_barrier_localvol_GBM(
                isCall,
                isContinuous,
                "DOWN-AND-OUT",
                x0,
                strike,
                tau,
                barrier,
                ir,
                dividend_yield,
                **kwargs,
            )
            self.BSObj = GBM_obj(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)
        elif flavor == "UP-AND-IN":
            self.barrierObj = single_barrier_localvol_GBM(
                isCall,
                isContinuous,
                "UP-AND-OUT",
                x0,
                strike,
                tau,
                barrier,
                ir,
                dividend_yield,
                **kwargs,
            )
            self.BSObj = GBM_obj(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)

    def price(self):
        if self.flavor in ["DOWN-AND-OUT", "UP-AND-OUT"]:
            return self.barrierObj.price()
        else:
            value = self.BSObj.price() - self.barrierObj.price()
            return value


class digital_localvol_base(single_asset_vol_base):
    """This is the local volatility base class for digital options"""

    def __init__(
        self, isCall, isCashOrNothing, x0, strike, tau, ir, dividend_yield, **kwargs
    ):
        super().__init__(isCall, x0, strike, tau, ir, dividend_yield, **kwargs)
        # kwargs = {
        #     k.upper(): v for k, v in kwargs.items()
        # }  # Convert all keys to upper case.
        self.isCashOrNothing = isCashOrNothing
        # self.other_params = kwargs
        self.isLog = dictGetAttr(self.other_params, "isLog", False)
        if not self.isLog:
            if self.isCashOrNothing:
                if self.isCall:
                    # f_up = customFunc("constant", a=1)  # np.exp(-self.ir * self.tau))
                    self.f_up = customFunc(
                        "EXP TIME", a=1, b=-self.ir * self.tau, c=self.ir
                    )
                    self.f_dn = customFunc("constant", a=0)
                    self.g = customFunc("Step", values=[0, 1], intervals=[self.strike])
                else:
                    self.f_up = customFunc("constant", a=0)
                    # f_dn = customFunc("constant", a=1)  # np.exp(-self.ir * self.tau))
                    self.f_dn = customFunc(
                        "EXP TIME", a=1, b=-self.ir * self.tau, c=self.ir
                    )
                    self.g = customFunc("Step", values=[1, 0], intervals=[self.strike])
            else:
                if self.isCall:
                    self.f_up = customFunc("linear", a=0, b=1)
                    self.f_dn = customFunc("constant", a=0)
                    left_functor = customFunc("constant", a=0)
                    right_functor = customFunc("linear", a=0, b=1)
                    self.g = customFunc(
                        "BI-PIECEWISE",
                        middle_point=self.strike,
                        left_functor=left_functor,
                        right_functor=right_functor,
                        side="Right",
                    )
                else:
                    self.f_up = customFunc("constant", a=0)
                    self.f_dn = customFunc("linear", a=0, b=1)
                    left_functor = customFunc("linear", a=0, b=1)
                    right_functor = customFunc("constant", a=0)
                    self.g = customFunc(
                        "BI-PIECEWISE",
                        middle_point=self.strike,
                        left_functor=left_functor,
                        right_functor=right_functor,
                        side="Right",
                    )
        else:
            raise NotImplementedError(
                "Log dynamics have not been implemented for the CEV model"
            )

    def price(self):
        if not self.isLog:
            return self.mesh(self.x0)
        else:
            return self.mesh(np.log(self.x0))


class digital_localvol_GBM(gbmMixin, digital_localvol_base):
    """Local vol price for Geometric Brownian motion
    Mixin class must come as the first parameter"""

    def __init__(
        self, isCall, isCashOrNothing, x0, strike, tau, ir, dividend_yield, **kwargs
    ):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """
        super().__init__(
            isCall, isCashOrNothing, x0, strike, tau, ir, dividend_yield, **kwargs
        )


class digital_localvol_CEV(cevMixin, digital_localvol_base):
    """Local vol price for Geometric Brownian motion
    Mixin class must come as the first parameter"""

    def __init__(
        self, isCall, isCashOrNothing, x0, strike, tau, ir, dividend_yield, **kwargs
    ):
        """Recommended additional parameters to be supplied:
        theta    :   theta=1/2: Crank-Nicolson; theta=1: fully implicit; theta=0: fully explicit
        M_high   :   upper bound in underlying prices
        M_low    :   lower bound in underlying prices
        m, n     :   discretization in underlying/time domain
        """
        super().__init__(
            isCall, isCashOrNothing, x0, strike, tau, ir, dividend_yield, **kwargs
        )
