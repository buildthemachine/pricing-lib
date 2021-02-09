"""
This is the test file for instruments.py.
Author:     Yufei Shen
Date:       12/21/2020
"""

import numpy as np
import scipy
import instruments
import logging
import unittest
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Test_bs_euroopt(unittest.TestCase):
    """Test the instruments.european_option pricing object functions as a correct wrapper
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
        self.bs_opt_1 = instruments.european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )
        # PDE solution with outright underlying prices
        self.bs_opt_2 = instruments.european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
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
        self.bs_opt_3 = instruments.european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
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
        self.cev_opt_1 = instruments.european_option(
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
        self.cev_opt_2 = instruments.european_option(
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

    def test_PriceEquivalence(self):
        self.assertTrue(
            np.fabs(self.bs_opt_2.Price() / self.bs_opt_1.Price() - 1) < self.eps
        )
        self.assertTrue(
            np.fabs(self.bs_opt_3.Price() / self.bs_opt_1.Price() - 1) < self.eps
        )
        self.assertTrue(
            np.fabs(self.cev_opt_2.Price() / self.cev_opt_1.Price() - 1) < self.eps
        )


class test_euro_GBM_digitalopt(unittest.TestCase):
    """Test the european_digital_option pricing object functions as a correct wrapper
    on its underlying pricing functions"""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.eps = 1e-2
        self.bsvol = 0.3
        self.bachvol = self.bsvol * self.underlying
        self.ir = 0.05
        self.dividend_yield = 0.02

        # Analytical solution for BS digital call option
        self.bs_digital_call_opt_1 = instruments.european_digital_option(
            isCall=True,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )

        # PDE solution for BS digital call option
        self.bs_digital_call_opt_2 = instruments.european_digital_option(
            isCall=True,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=800,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 18,
            Mlow=0,
        )

        # Analytical solution for BS digital put option
        self.bs_digital_put_opt_1 = instruments.european_digital_option(
            isCall=False,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )

        # PDE solution for BS digital put option
        self.bs_digital_put_opt_2 = instruments.european_digital_option(
            isCall=False,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=500,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 18,
            Mlow=0,
        )

        # Analytical solution for BS digital call option: Asset or Nothing
        self.bs_digital_call_opt_3 = instruments.european_digital_option(
            isCall=True,
            isCashOrNothing=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )

        # PDE solution for BS digital call option: Asset or Nothing
        self.bs_digital_call_opt_4 = instruments.european_digital_option(
            isCall=True,
            isCashOrNothing=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=500,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 18,
            Mlow=0,
        )

        # Analytical solution for BS digital put option: Asset or Nothing
        self.bs_digital_put_opt_3 = instruments.european_digital_option(
            isCall=False,
            isCashOrNothing=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )

        # PDE solution for BS digital put option: Asset or Nothing
        self.bs_digital_put_opt_4 = instruments.european_digital_option(
            isCall=False,
            isCashOrNothing=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=800,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 18,
            Mlow=0,
        )

    def test_PriceEquivalence(self):
        self.assertTrue(
            np.fabs(
                self.bs_digital_call_opt_2.Price() / self.bs_digital_call_opt_1.Price()
                - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.bs_digital_put_opt_2.Price() / self.bs_digital_put_opt_1.Price()
                - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.bs_digital_call_opt_4.Price() / self.bs_digital_call_opt_3.Price()
                - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.bs_digital_put_opt_4.Price() / self.bs_digital_put_opt_3.Price()
                - 1
            )
            < self.eps
        )


class test_euro_CEV_digitalopt(unittest.TestCase):
    """Test the CEV digital option pricing.
    The analytical solution to CEV option is only in an approximate sense, thereby eps is chosen
    to be 0.05"""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.eps = 5e-2
        self.ir = 0.05
        self.dividend_yield = 0.02
        self.cev_p = 0.5
        self.cev_lamda = 0.6

        # Analytical call solution for CEV
        self.cev_digital_call_opt_1 = instruments.european_digital_option(
            isCall=True,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )

        # PDE call solution for CEV
        self.cev_digital_call_opt_2 = instruments.european_digital_option(
            isCall=True,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=800,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 18,
            Mlow=0,
        )

        # Analytical put solution for CEV
        self.cev_digital_put_opt_1 = instruments.european_digital_option(
            isCall=False,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["ANALYTICAL", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
        )

        # PDE put solution for CEV
        self.cev_digital_put_opt_2 = instruments.european_digital_option(
            isCall=False,
            isCashOrNothing=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=800,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

    def test_PriceEquivalence(self):
        self.assertTrue(
            np.fabs(
                self.cev_digital_call_opt_2.Price()
                / self.cev_digital_call_opt_1.Price()
                - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.cev_digital_put_opt_2.Price() / self.cev_digital_put_opt_1.Price()
                - 1
            )
            < self.eps
        )


class test_euro_barrieropt_asymptotics(unittest.TestCase):
    """Test Euro barrier option pricing functions
    When the barrier levels are extremely high/low, the 'out' barrier option prices should
    approach the underlying BS prices, while the 'in' barrier option prices should be 0."""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.ir = 0.05
        self.dividend_yield = 0.02
        self.bsvol = 0.3
        self.eps = 1e-5

        # Analytical solution for continuous European call barrier option:
        self.bs_cont_call_dni_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="DOWN-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=1e-8,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for continuous European call barrier option:
        self.bs_cont_call_dno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="DOWN-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=1e-8,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for Black-Scholes
        self.bs_call_1 = instruments.european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for continuous European call barrier option:
        self.bs_cont_call_uno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=1e6,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for continuous European call barrier option:
        self.bs_cont_call_uni_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=1e5,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for continuous European put barrier option:
        self.bs_cont_put_uno_1 = instruments.european_single_barrier_option(
            isCall=False,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=1e6,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for continuous European put barrier option:
        self.bs_cont_put_uni_1 = instruments.european_single_barrier_option(
            isCall=False,
            isContinuous=True,
            flavor="UP-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=1e4,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        # Analytical solution for Black-Scholes
        self.bs_put_1 = instruments.european_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

    def test_asymptotics(self):
        """If set the barrier level to be extreme high/low, then the up/down-and-out
        options will have the same price as Black-Scholes"""
        self.assertTrue(self.bs_cont_call_dni_1.Price() < self.eps)
        self.assertTrue(
            np.fabs(self.bs_cont_call_dno_1.Price() / self.bs_call_1.Price() - 1)
            < self.eps
        )
        self.assertTrue(
            np.fabs(self.bs_cont_call_uno_1.Price() / self.bs_call_1.Price() - 1)
            < self.eps
        )
        self.assertTrue(self.bs_cont_call_uni_1.Price() < self.eps)
        self.assertTrue(self.bs_cont_put_uni_1.Price() < self.eps)
        self.assertTrue(
            np.fabs(self.bs_cont_put_uno_1.Price() / self.bs_put_1.Price() - 1)
            < self.eps
        )


class test_gbm_euro_barrieropt_priceEquivalence(unittest.TestCase):
    """Test Euro barrier option pricing functions
    The barrier option analytical price should be very close to PDE results."""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.ir = 0.05
        self.dividend_yield = 0.02
        self.bsvol = 0.3
        self.eps = 5e-4

        # PDE solution for continuous European call barrier option:
        self.barrier = 1e-8
        self.pde_cont_call_dno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="DOWN-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=self.barrier,
        )

        self.bs_cont_call_dno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="DOWN-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        self.barrier = 1000
        self.pde_cont_call_uno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.barrier,
            Mlow=0,
        )

        self.bs_cont_call_uno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        self.pde_cont_call_uni_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="up-and-in",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.barrier,
            Mlow=0,
        )

        self.bs_cont_call_uni_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        self.barrier = 150
        self.pde_cont_call_uno_2 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.barrier,
            Mlow=0,
        )

        self.bs_cont_call_uno_2 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        self.pde_cont_call_uni_2 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="up-and-in",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.barrier,
            Mlow=0,
        )

        self.bs_cont_call_uni_2 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

    def test_priceEquivalence(self):
        self.assertTrue(
            np.fabs(
                self.pde_cont_call_dno_1.Price() / self.bs_cont_call_dno_1.Price() - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.pde_cont_call_uno_1.Price() / self.bs_cont_call_uno_1.Price() - 1
            )
            < self.eps
            or np.fabs(
                self.pde_cont_call_uno_1.Price() - self.bs_cont_call_uno_1.Price()
            )
            < 0.02
        )
        self.assertTrue(
            np.fabs(
                self.pde_cont_call_uni_1.Price() / self.bs_cont_call_uni_1.Price() - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.pde_cont_call_uno_2.Price() / self.bs_cont_call_uno_2.Price() - 1
            )
            < self.eps
            or np.fabs(
                self.pde_cont_call_uno_2.Price() - self.bs_cont_call_uno_2.Price()
            )
            < 0.02
        )
        self.assertTrue(
            np.fabs(
                self.pde_cont_call_uni_2.Price() / self.bs_cont_call_uni_2.Price() - 1
            )
            < self.eps
        )


class test_cev_euro_barrieropt_asymptotics(unittest.TestCase):
    """Test Euro barrier option pricing functions
    The barrier option analytical price should be very close to PDE results."""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.eps = 1e-4
        self.ir = 0.09
        self.dividend_yield = 0.02
        self.cev_p = 0.5
        self.cev_lamda = 0.6

        self.bs_cev_call_1 = instruments.european_option(
            isCall=True,
            isContinuous=True,
            flavor="DOWN-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["analytical", "CEV"],
            p=self.cev_p,
            lamda=self.cev_lamda,
        )

        # PDE solution for continuous European call barrier option:
        self.barrier = 1e-8
        self.pde_cont_call_dno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="DOWN-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "CEV"],
            p=self.cev_p,
            lamda=self.cev_lamda,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=self.barrier,
        )

        # PDE solution for continuous European call barrier option:
        self.barrier = 1000
        self.pde_cont_call_uno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "CEV"],
            p=self.cev_p,
            lamda=self.cev_lamda,
            interp="cubic spline",
            M=300,  # spatial grids
            N=200,  # time grids
            Mhigh=self.barrier,
            Mlow=0,
        )

    def test_asymptotics(self):
        self.assertTrue(
            np.fabs(self.pde_cont_call_dno_1.Price() / self.bs_cev_call_1.Price() - 1)
            < self.eps
        )
        self.assertTrue(
            np.fabs(self.pde_cont_call_uno_1.Price() / self.bs_cev_call_1.Price() - 1)
            < self.eps
        )


class test_discrete_barrieropt_gbm(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.ir = 0.05
        self.dividend_yield = 0.02
        self.bsvol = 0.31
        self.eps = 1e-2

        self.barrier = 500
        self.bs_disc_call_uno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=False,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
            barrier_obs_freq="hourly",
        )

        self.bs_cont_call_uno_1 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=True,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
        )

        self.barrier = 300
        self.bs_disc_call_uno_2 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=False,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
            barrier_obs_freq="monthly",
        )

        self.bs_disc_call_uno_2_pde = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=False,
            flavor="UP-AND-OUT",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            timeObsFreq="monthly",
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.barrier * 3,
            Mlow=0,
        )

        self.bs_disc_call_uni_2 = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=False,
            flavor="UP-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["analytical", "GBM"],
            vol=self.bsvol,
            barrier_obs_freq="monthly",
        )

        self.bs_disc_call_uni_2_pde = instruments.european_single_barrier_option(
            isCall=True,
            isContinuous=False,
            flavor="UP-AND-IN",
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            barrier=self.barrier,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            timeObsFreq="monthly",
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.barrier * 3,
            Mlow=0,
        )

    def test_asymptotics(self):
        self.assertTrue(
            np.fabs(
                self.bs_disc_call_uno_1.Price() / self.bs_cont_call_uno_1.Price() - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.bs_disc_call_uno_2.Price() / self.bs_disc_call_uno_2_pde.Price()
                - 1
            )
            < self.eps
        )
        self.assertTrue(
            np.fabs(
                self.bs_disc_call_uni_2.Price() / self.bs_disc_call_uni_2_pde.Price()
                - 1
            )
            < self.eps
        )


class test_american_put_carr(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 1
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.ir = 0.1
        self.dividend_yield = 0.0
        self.bsvol = 0.3
        self.eps = 1e-2

        self.american_opt_put_randomization = instruments.american_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=0,
            Engine=["ANALYTICAL", "GBM"],
            vol=self.bsvol,
            richardson_order=5,
        )

    def test_price(self):
        self.assertTrue(
            np.fabs(self.american_opt_put_randomization.Price() - 8.3378) < self.eps
        )
        prices = {}
        s_low, s_high = 50, 150
        spots = np.linspace(s_low, s_high, 500)
        fig = plt.figure(figsize=(15, 10))
        for order in range(1, 5):
            prices[order] = []
            for x0 in spots:
                self.american_opt_put_randomization = (
                    instruments.american_vanilla_option(
                        isCall=False,
                        strike=self.strike,
                        underlying=x0,
                        expiration=self.tau,
                        ir=self.ir,
                        dividend_yield=0,
                        Engine=["ANALYTICAL", "GBM"],
                        vol=self.bsvol,
                        richardson_order=order,
                    )
                )
                prices[order].append(self.american_opt_put_randomization.Price())
            plt.plot(spots, prices[order], label=f"order={order}")
        euro_price = []
        for x0 in spots:
            # Analytical solution for Black-Scholes
            self.bs_put = instruments.european_option(
                isCall=False,
                strike=self.strike,
                underlying=x0,
                expiration=self.tau,
                ir=self.ir,
                dividend_yield=0,
                Engine=["analytical", "GBM"],
                vol=self.bsvol,
            )
            euro_price.append(self.bs_put.Price())
        plt.plot(spots, euro_price, label="European")
        plt.plot(spots, np.maximum(self.strike - spots, 0), label="payoff")
        plt.xlim((s_low, s_high))
        plt.ylim((0, self.strike - s_low))
        plt.legend(loc="upper right")
        # plt.show()


class test_american_option(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.ir = 0.05
        self.dividend_yield = 0.0
        self.bsvol = 0.31
        self.cev_p = 0.5
        self.cev_lamda = 0.6
        self.eps = 1e-2

        self.american_opt_call_1 = instruments.american_vanilla_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

        self.euro_opt_call_1 = instruments.european_option(
            isCall=True,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

        self.american_opt_put_1 = instruments.american_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

        self.euro_opt_put_1 = instruments.european_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

        self.american_opt_put_2 = instruments.american_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

        self.euro_opt_put_2 = instruments.european_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.underlying * 10,
            Mlow=0,
        )

    def test_asymptotics(self):
        self.assertTrue(
            np.fabs(self.american_opt_call_1.Price() / self.euro_opt_call_1.Price() - 1)
            < self.eps
        )
        self.assertTrue(self.american_opt_put_1.Price() > self.euro_opt_put_1.Price())
        self.assertTrue(self.american_opt_put_2.Price() > self.euro_opt_put_2.Price())


# The following test is quite time consuming therefore better to comment out
class test_bermudan_option(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = np.linspace(50, 200, 100)
        self.strike = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.ir = 0.05
        self.dividend_yield = 0.0
        self.bsvol = 0.31
        self.cev_p = 0.5
        self.cev_lamda = 0.6
        self.eps = 1e-2

        self.bermudan_opt_put_annual_cev = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            interp="cubic spline",
            timeObsFreq="annually",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_month_cev = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            interp="cubic spline",
            timeObsFreq="monthly",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_week_cev = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            interp="cubic spline",
            timeObsFreq="weekly",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_day_cev = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            interp="cubic spline",
            timeObsFreq="daily",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.american_opt_put_cev = instruments.american_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            interp="cubic spline",
            # timeObsFreq="monthly",
            M=300,  # spatial grids
            N=3000,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.euro_opt_put_cev = instruments.european_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "CEV"],
            lamda=self.cev_lamda,
            p=self.cev_p,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_annual = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            timeObsFreq="annually",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_month = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            timeObsFreq="monthly",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_week = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            timeObsFreq="weekly",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.bermudan_opt_put_day = instruments.bermudan_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            timeObsFreq="daily",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.american_opt_put = instruments.american_vanilla_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            interp="cubic spline",
            # timeObsFreq="monthly",
            M=300,  # spatial grids
            N=3000,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

        self.euro_opt_put = instruments.european_option(
            isCall=False,
            strike=self.strike,
            underlying=self.underlying,
            expiration=self.tau,
            Engine=["PDE", "GBM"],
            vol=self.bsvol,
            ir=self.ir,
            dividend_yield=self.dividend_yield,
            interp="cubic spline",
            M=300,  # spatial grids
            N=300,  # time grids
            Mhigh=self.strike * 10,
            Mlow=0,
        )

    def test_asymptotics(self):
        annual_price = self.bermudan_opt_put_annual.Price()
        month_price = self.bermudan_opt_put_month.Price()
        week_price = self.bermudan_opt_put_week.Price()
        day_price = self.bermudan_opt_put_day.Price()

        annual_price_cev = self.bermudan_opt_put_annual_cev.Price()
        month_price_cev = self.bermudan_opt_put_month_cev.Price()
        week_price_cev = self.bermudan_opt_put_week_cev.Price()
        day_price_cev = self.bermudan_opt_put_day_cev.Price()

        self.assertTrue(np.all(self.euro_opt_put.Price() < annual_price))
        self.assertTrue(np.all(annual_price < month_price))
        self.assertTrue(np.all(month_price < week_price))
        self.assertTrue(np.all(week_price < day_price))
        self.assertTrue(np.all(day_price < self.american_opt_put.Price()))

        self.assertTrue(np.all(self.euro_opt_put_cev.Price() < annual_price_cev))
        self.assertTrue(np.all(annual_price_cev < month_price_cev))
        self.assertTrue(np.all(month_price_cev < week_price_cev))
        self.assertTrue(np.all(week_price_cev < day_price_cev))
        self.assertTrue(np.all(day_price_cev < self.american_opt_put_cev.Price()))


if __name__ == "__main__":
    unittest.main()
