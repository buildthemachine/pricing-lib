"""
This is the test file for instruments.py.
Author:     Yufei Shen
Date:       12/21/2020
"""

import numpy as np
import scipy
import LocalVol.lv_utils
from instruments import european_option
from abc import ABCMeta
import logging
import unittest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Test_bs_euroopt(unittest.TestCase):
    """Test the european_option pricing object functions as a correct wrapper
    on its underlying pricing functions"""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.eps = 1e-4
        self.bsvol = 0.3
        self.bachvol = self.bsvol * self.underlying
        self.ir = 0.09
        self.dividend_yield = 0.02
        self.cev_p = 0.5
        self.cev_lamda = 0.6

        # Analytical solution for BS
        self.bs_opt_1 = european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "BS"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )
        # PDE solution with outright underlying prices
        self.bs_opt_2 = european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "BS"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 13,
            Mlow=0,
        )
        # PDE solution with logarithm underlying prices
        self.bs_opt_3 = european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "BS"],
            isLog=True,
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mlow=np.log(self.underlying / 13),
            Mhigh=np.log(self.underlying * 13),
        )

        # Analytical solution for CEV
        self.cev_opt_1 = european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )
        # PDE solution with outright underlying prices
        self.cev_opt_2 = european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=500,  # spatial grids
            N=200,  # time grids
            Mlow=self.underlying / 10,
            Mhigh=self.underlying * 5,
        )

    def test_priceEquivalence(self):
        self.assertTrue(
            np.fabs(self.bs_opt_2.Price() / self.bs_opt_1.Price() - 1) < self.eps
        )
        self.assertTrue(
            np.fabs(self.bs_opt_3.Price() / self.bs_opt_1.Price() - 1) < self.eps
        )
        self.assertTrue(
            np.fabs(self.cev_opt_2.Price() / self.cev_opt_1.Price() - 1) < self.eps
        )
