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
from abc import ABCMeta, abstractmethod
import logging
from Utils.other_utils import dictGetAttr


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class instrument_base(metaclass=ABCMeta):
    """Define a virtual base class for financial instruments"""

    @abstractmethod
    def __init__(
        self, vol_or_price, isCall, strike, underlying, expiration, Engine, **kwargs
    ):
        self.vol_or_price = vol_or_price  # 0: return implied vol; 1: return price
        self.isCall = isCall
        self.strike = strike
        self.x0 = underlying
        self.tau = expiration
        self._Engine = Engine
        kwargs = {
            k.upper(): v for k, v in kwargs.items()
        }  # Convert all keys to upper case.
        self.other_params = kwargs

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
            self._Engine = val_dict
        else:
            raise ValueError(
                f"The specificed engine is not supported. The supported engines are: {self.engine_dict}"
            )

    def Price(self):
        return self._priceObj.price()

    def impVol(self):
        return self._priceObj.impliedVol()


class european_digital_option(instrument_base):
    """Define European options class, with digital (binary) payouts"""

    def __init__(
        self, isCall, isCashOrNothing, strike, underlying, expiration, Engine, **kwargs
    ):
        super().__init__(self, isCall, strike, underlying, expiration, Engine, **kwargs)
        self.isCashOrNothing = isCashOrNothing
        self.engine_dict = {
            "ANALYTICAL": ["GBM", "CEV", "SABR"],
            "PDE": ["GBM", "CEV"],
            "MONTE CARLO": ["TBD"],
        }  # The suppported engines vary by each instrument.
        if (self._Engine[0] not in self.engine_dict) or (
            self._Engine[1] not in self.engine_dict[self._Engine[0]]
        ):
            logger.warning(
                "The specified calculation engine is not supported!\nThe currently supported engines are:"
            )
            logger.warning(self.engine_dict)
            raise KeyError

        # Obtaining underlying volatility object
        if self.Engine[0] == "ANALYTICAL":
            if self.Engine[1] == "GBM":
                self._priceObj = Utils.vanilla_utils.GBM_digital_obj(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = Utils.vanilla_utils.CEV_digital_obj(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    **self.other_params,
                )
            elif self.Engine[1] == "SABR":
                self._priceObj = Utils.vanilla_utils.SABR_digital_obj(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    **self.other_params,
                )
        elif self.Engine[0] == "PDE":
            if self._Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.digital_localvol_GBM(
                    self.isCall,
                    self.isCashOrNothing,
                    self.x0,
                    self.strike,
                    self.tau,
                    ir=dictGetAttr(self.other_params, "ir", None),
                    dividend_yield=dictGetAttr(
                        self.other_params, "dividend_yield", None
                    ),
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
                    ir=dictGetAttr(self.other_params, "ir", None),
                    dividend_yield=dictGetAttr(
                        self.other_params, "dividend_yield", None
                    ),
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
        Engine=["ANALYTICAL", "GBM"],
        **kwargs,
    ):
        super().__init__(self, isCall, strike, underlying, expiration, Engine, **kwargs)
        self.engine_dict = {
            "ANALYTICAL": ["GBM", "CEV", "DISPLACED GBM", "SABR"],
            "PDE": ["GBM", "CEV"],
            "MONTE CARLO": ["TBD"],
        }  # The suppported engines vary by each instrument.
        if (self._Engine[0] not in self.engine_dict) or (
            self._Engine[1] not in self.engine_dict[self._Engine[0]]
        ):
            logger.warning(
                "The specified calculation engine is not supported!\nThe currently supported engines are:"
            )
            logger.warning(self.engine_dict)
            raise KeyError

        # Obtaining underlying volatility object
        if self.Engine[0] == "ANALYTICAL":
            if self.Engine[1] == "GBM":
                self._priceObj = Utils.vanilla_utils.GBM_obj(
                    self.isCall, self.x0, self.strike, self.tau, **self.other_params
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = Utils.vanilla_utils.CEV_obj(
                    self.isCall, self.x0, self.strike, self.tau, **self.other_params
                )
            elif self.Engine[1] == "SABR":
                self._priceObj = Utils.vanilla_utils.sabr_black_vol(
                    self.isCall, self.x0, self.strike, self.tau, **self.other_params
                )
        elif self.Engine[0] == "PDE":
            if self._Engine[1] == "GBM":
                self._priceObj = LocalVol.lv.vanilla_localvol_GBM(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    ir=dictGetAttr(self.other_params, "ir", None),
                    dividend_yield=dictGetAttr(
                        self.other_params, "dividend_yield", None
                    ),
                    **self.other_params,
                )
            elif self.Engine[1] == "CEV":
                self._priceObj = LocalVol.lv.vanilla_localvol_CEV(
                    self.isCall,
                    self.x0,
                    self.strike,
                    self.tau,
                    isLog=False,
                    ir=dictGetAttr(self.other_params, "ir", None),
                    dividend_yield=dictGetAttr(
                        self.other_params, "dividend_yield", None
                    ),
                    **self.other_params,
                )
        elif self.Engine[0] == "MONTE CARLO":
            raise NotImplementedError("Monte Carlo engine has yet to be implemented")

    def greeks(self):
        """TODO: definition for calculating the greeks of the product"""
        raise NotImplementedError("Greek has not been implemented.")
