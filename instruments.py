"""
This code implements the different utility functions of local volatility.
Author:     Yufei Shen
Date:       12/20/2020
# pricing-lib

"""

import numpy as np
import scipy
import LocalVol.lv
import Utils.vanilla_utils
import Utils.exotic_utils
from abc import ABCMeta, abstractmethod
import logging
from Utils.other_utils import dictGetAttr


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class instrument_base(metaclass=ABCMeta):
    """Define a virtual base class for financial instruments"""

    @abstractmethod
    def __init__(
        self,
        # vol_or_price,
        isCall,
        strike,
        underlying,
        expiration,
        ir,
        dividend_yield,
        Engine,
        engine_dict,
        **kwargs,
    ):
        # self.vol_or_price = vol_or_price  # 0: return implied vol; 1: return price
        self.isCall = isCall
        self.strike = strike
        self.x0 = underlying
        self.tau = expiration
        self.ir = ir
        self.dividend_yield = dividend_yield
        self.engine_dict = engine_dict
        self.Engine = Engine
        kwargs = {
            k.upper(): v for k, v in kwargs.items()
        }  # Convert all keys to upper case.
        self.other_params = kwargs
        if (self.Engine[0] not in self.engine_dict) or (
            self.Engine[1] not in self.engine_dict[self.Engine[0]]
        ):
            raise KeyError(
                f"The specified calculation engine is not supported!\n The currently supported engines are:{self.engine_dict}"
            )

    @property
    def Engine(self):
        if "Engine" in instrument_base.__dict__:
            return self._Engine
        else:
            raise NotImplementedError(
                "The computation engine has not been set!\nPlease provide definition for self.Engine."
            )

    @Engine.setter
    def Engine(self, val_list):
        """val_list[0] is the category of underlying calculation engine, while the rest of the list
        contains additional parameters"""
        value = val_list[0].upper()
        if value in self.engine_dict:
            self._Engine = [s.upper() for s in val_list]
        else:
            raise ValueError(
                f"The specificed engine is not supported. The supported engines are: {self.engine_dict}"
            )

    def Price(self):
        return self._priceObj.price()

    def impVol(self):
        return self._priceObj.impliedVol()


class european_single_barrier_option(instrument_base):
    """Define European options class, with barrier flavors.
    It still allows for analytical solutions given in Pg 604 in Hull."""

    def __init__(
        self,
        isCall,
        isContinuous,
        flavor,
        strike,
        underlying,
        expiration,
        ir,
        dividend_yield,
        barrier,
        Engine,
        **kwargs,
    ):
        super().__init__(
            isCall,
            strike,
            underlying,
            expiration,
            ir,
            dividend_yield,
            Engine,
            engine_dict={
                "ANALYTICAL": ["GBM"],
                "PDE": ["GBM", "CEV"],
                "MONTE CARLO": ["TBD"],
            },  # The suppported engines vary by each instrument.
            **kwargs,
        )
        self.isContinuous = isContinuous
        self.flavor = flavor.upper()
        self.barrier = barrier
        if self.flavor not in [
            "UP-AND-IN",
            "UP-AND-OUT",
            "DOWN-AND-IN",
            "DOWN-AND-OUT",
        ]:
            raise KeyError(
                f"The flavor {self.flavor} barrier option is currently not supported!"
            )

        # Obtaining underlying volatility object
        if self.Engine[0] == "ANALYTICAL":
            if self.Engine[1] == "GBM":
                self._priceObj = Utils.exotic_utils.GBM_barrier_obj(
                    self.isCall,
                    self.isContinuous,
                    self.flavor,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    self.barrier,
                    **self.other_params,
                )
        elif self.Engine[0] == "PDE":
            if self.Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.single_barrier_localvol_GBM_Facade(
                    self.isCall,
                    self.isContinuous,
                    self.flavor,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.barrier,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = LocalVol.lv.single_barrier_localvol_CEV_Facade(
                    self.isCall,
                    self.isContinuous,
                    self.flavor,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.barrier,
                    isLog=False,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "MONTE CARLO":
            raise NotImplementedError("Monte Carlo engine has yet to be implemented")


class european_digital_option(instrument_base):
    """Define European options class, with digital (binary) payouts"""

    def __init__(
        self,
        isCall,
        isCashOrNothing,
        strike,
        underlying,
        expiration,
        ir,
        dividend_yield,
        Engine,
        **kwargs,
    ):
        super().__init__(
            isCall,
            strike,
            underlying,
            expiration,
            ir,
            dividend_yield,
            Engine,
            engine_dict={
                "ANALYTICAL": ["GBM", "CEV"],
                "PDE": ["GBM", "CEV"],
                "MONTE CARLO": ["TBD"],
            },  # The suppported engines vary by each instrument.
            **kwargs,
        )
        self.isCashOrNothing = isCashOrNothing

        # Obtaining underlying volatility object
        if self.Engine[0] == "ANALYTICAL":
            if self.Engine[1] == "GBM":
                self._priceObj = Utils.vanilla_utils.GBM_digital_obj(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = Utils.vanilla_utils.CEV_digital_obj(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "SABR":
                self._priceObj = Utils.vanilla_utils.SABR_digital_obj(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "PDE":
            if self.Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.digital_localvol_GBM(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = LocalVol.lv.digital_localvol_CEV(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    isLog=False,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "MONTE CARLO":
            raise NotImplementedError("Monte Carlo engine has yet to be implemented")


class american_vanilla_option(instrument_base):
    """Define American option class, where early exercise is available throughout maturity T"""

    def __init__(
        self,
        isCall,
        strike,
        underlying,
        expiration,
        ir,
        dividend_yield,
        Engine=["PDE", "GBM"],
        **kwargs,
    ):
        super().__init__(
            isCall,
            strike,
            underlying,
            expiration,
            ir,
            dividend_yield,
            Engine,
            engine_dict={
                "ANALYTICAL": ["GBM"],
                "PDE": ["GBM", "CEV"],
                "MONTE CARLO": ["TBD"],
            },  # The suppported engines vary by each instrument.
            **kwargs,
        )
        # Obtaining underlying volatility object
        if self.Engine[0] == "ANALYTICAL":
            self._priceObj = Utils.vanilla_utils.GBM_randomization(
                self.isCall,
                self.x0,
                self.strike,
                self.tau,
                ir=self.ir,
                **self.other_params,
            )
        elif self.Engine[0] == "PDE":
            if self.Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.american_vanilla_localvol_GBM(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = LocalVol.lv.american_vanilla_localvol_CEV(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    isLog=False,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "MONTE CARLO":
            raise NotImplementedError("Monte Carlo engine has yet to be implemented")


class bermudan_vanilla_option(instrument_base):
    """Define Bermudan option class, where early exercise is possible on a pre-defined time grid till maturity T"""

    def __init__(
        self,
        isCall,
        strike,
        underlying,
        expiration,
        ir,
        dividend_yield,
        Engine=["PDE", "GBM"],
        **kwargs,
    ):
        super().__init__(
            isCall,
            strike,
            underlying,
            expiration,
            ir,
            dividend_yield,
            Engine,
            engine_dict={
                # "ANALYTICAL": ["GBM", "CEV", "DISPLACED GBM", "SABR"],
                "PDE": ["GBM", "CEV"],
                "MONTE CARLO": ["TBD"],
            },  # The suppported engines vary by each instrument.
            **kwargs,
        )
        # Obtaining underlying volatility object
        if self.Engine[0] == "PDE":
            if self.Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.bermudan_vanilla_localvol_GBM(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = LocalVol.lv.bermudan_vanilla_localvol_CEV(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    isLog=False,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "MONTE CARLO":
            raise NotImplementedError("Monte Carlo engine has yet to be implemented")


class european_option(instrument_base):
    """Define European options class"""

    def __init__(
        self,
        isCall,
        strike,
        underlying,
        expiration,
        ir,
        dividend_yield,
        Engine=["ANALYTICAL", "GBM"],
        **kwargs,
    ):
        super().__init__(
            isCall,
            strike,
            underlying,
            expiration,
            ir,
            dividend_yield,
            Engine,
            engine_dict={
                "ANALYTICAL": ["GBM", "CEV", "DISPLACED GBM", "SABR"],
                "PDE": ["GBM", "CEV"],
                "MONTE CARLO": ["TBD"],
            },  # The suppported engines vary by each instrument.
            **kwargs,
        )

        # Obtaining underlying volatility object
        if self.Engine[0] == "ANALYTICAL":
            if self.Engine[1] == "GBM":
                self._priceObj = Utils.vanilla_utils.GBM_obj(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = Utils.vanilla_utils.CEV_obj(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "SABR":
                self._priceObj = Utils.vanilla_utils.sabr_black_vol(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    self.ir,
                    self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "PDE":
            if self.Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.vanilla_localvol_GBM(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = LocalVol.lv.vanilla_localvol_CEV(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    isLog=False,
                    ir=self.ir,
                    dividend_yield=self.dividend_yield,
                    **self.other_params,
                )
        elif self.Engine[0] == "MONTE CARLO":
            raise NotImplementedError("Monte Carlo engine has yet to be implemented")

    def greeks(self):
        """TODO: definition for calculating the greeks of the product"""
        raise NotImplementedError("Greek has not been implemented.")
