"""
This code implements the different utility functions of local volatility.
Author:     Yufei Shen
Date:       11/17/2020
"""

from abc import ABCMeta


class vol_base(metaclass=ABCMeta):
    """Define the abstract base interface of local volatility"""

    @abstractmethod
    def Price(self):
        pass

    @property
    @abstractmethod
    def Engine(self):
        pass


class localvol(vol_base, engineObj):
    """This is the local volatility class"""

    def __init__(self):
        # TODO
        self.supportedEngine = {"Analytical"    : ["CEV", "DLN", "BLACK"],  # Only vanilla models
                                "PDE"           : []
                                "Monte Carlo"   : []
                                }
        self._engineObj = engineObj

    @property
    def Engine(self, engineName):
        if (engineName in self.supportedEngine):
            return self._engineObj(engineName)


    def Price(self):
        #TODO: define price function





