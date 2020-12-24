"""
This is the test file for LocalVol.lv_utils.py.
Author:     Yufei Shen
Date:       12/21/2020
"""

import unittest
import numpy as np
from scipy.stats import norm
import logging
import LocalVol.lv_utils
from Utils.vanilla_utils import (
    customFunc,
    undiscBachelierFwd,
    BSOpt,
    CEVSpot,
    undiscBSOptFwd,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Test_Mesh(unittest.TestCase):
    """Test the Mesh class is implemented correctly to solve the backward induction
    problem in Feynman-Kac PDE."""

    @classmethod
    def setUpClass(cls):
        logger.info("Setting up class object...")

    @classmethod
    def tearDownClass(cls):
        logger.info("Tearing down class object...")

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.tau = 10
        self.underlying = 100
        self.theta = 0.5  # Theta controls explicit/implicit/Crank-Nicolson
        self.eps = 1e-4
        self.bsvol = 0.3
        self.bachvol = self.bsvol * self.underlying
        self.strike = self.underlying
        self.ir = 0.09
        self.dividend_yield = 0.02
        self.cev_p = 0.5
        self.cev_lamda = 0.6
        # Construct Black's model:
        # dS(t)=\sigma S(t)dW(t)
        self.mesh_BS_outright = LocalVol.lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=0,
            Mhigh=1000,
            interp="cubic spline",
            EquidistantGrid=True,
            m=400,
            n=300,
            r=customFunc("constant", a=0),
            mu=customFunc("constant", a=0),
            # b is BS volatility
            sigma=customFunc("linear", a=0, b=self.bsvol),
            f_up=customFunc("RELU", a=-self.strike, b=1),
            f_dn=customFunc("constant", a=0),
            g=customFunc("RELU", a=-self.strike, b=1),
        )
        # Construct Black's model with x=ln(S)
        # dx(t) = dln(S)=dS(t)/S-1/2*dS^2/S^2
        #       = -1/2\sigma^2 dt+\sigma dW(t)
        self.mesh_BS_log = LocalVol.lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=np.log(self.underlying / 10),
            Mhigh=np.log(self.underlying * 10),
            interp="cubic spline",
            EquidistantGrid=True,
            m=300,
            n=300,
            r=customFunc("constant", a=0),
            mu=customFunc("constant", a=-0.5 * self.bsvol ** 2),
            sigma=customFunc("constant", a=self.bsvol),
            f_up=customFunc("Exp RELU", a=1, b=1, c=-self.strike),
            f_dn=customFunc("constant", a=0),
            g=customFunc("Exp RELU", a=1, b=1, c=-self.strike),
        )
        # Construct Black-Scholes model:
        # dS(t)=(r-a)S(t)dt + \sigma S(t)dW(t)
        self.mesh_BS_outright_ir = LocalVol.lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=0,
            Mhigh=500,
            interp="cubic spline",
            EquidistantGrid=True,
            m=300,
            n=300,
            r=customFunc("constant", a=self.ir),
            mu=customFunc("linear", a=0, b=self.ir - self.dividend_yield),
            # b is BS volatility
            sigma=customFunc("linear", a=0, b=self.bsvol),
            f_up=customFunc(
                "Exp Diff",
                isCall=True,
                k=self.strike,
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            ),
            f_dn=customFunc("constant", a=0),
            g=customFunc("RELU", a=-self.strike, b=1),
        )
        # Construct Black-Scholes's model with x=ln(S)
        # dx(t) = dln(S)=dS(t)/S-1/2*dS^2/S^2
        #       = (r-a-1/2\sigma^2) dt+\sigma dW(t)
        self.mesh_BS_log_ir = LocalVol.lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=np.log(self.underlying / 10),
            Mhigh=np.log(self.underlying * 10),
            interp="cubic spline",
            EquidistantGrid=True,
            m=500,
            n=400,
            r=customFunc("constant", a=self.ir),
            mu=customFunc(
                "constant", a=self.ir - self.dividend_yield - 0.5 * self.bsvol ** 2
            ),
            sigma=customFunc("constant", a=self.bsvol),
            # f_up=customFunc("Exp RELU", a=1, b=1, c=-self.strike),
            f_up=customFunc(
                "Exp Diff 2",
                isCall=True,
                k=np.log(self.strike),
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            ),
            f_dn=customFunc("constant", a=0),
            g=customFunc("Exp RELU", a=1, b=1, c=-self.strike),
        )

        # Construct Bachelier model
        # dS(t) = \sigma dW(t)
        self.mesh_Bache_outright = LocalVol.lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=-300,
            Mhigh=500,
            interp="cubic spline",
            EquidistantGrid=True,
            m=600,
            n=400,
            r=customFunc("constant", a=0),
            mu=customFunc("constant", a=0),
            # a is Bachelier volatility
            sigma=customFunc("constant", a=self.bachvol),
            f_up=customFunc("RELU", a=-self.strike, b=1),
            f_dn=customFunc("constant", a=0),
            g=customFunc("RELU", a=-self.strike, b=1),
        )
        # Construct CEV model
        # dS(t) = (r-a)S(t)dt+\lambda S^p dW(t)
        self.mesh_CEV = LocalVol.lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=self.underlying / 10,
            Mhigh=self.underlying * 5,
            interp="cubic spline",
            EquidistantGrid=True,
            m=500,
            n=200,
            r=customFunc("constant", a=self.ir),
            mu=customFunc("linear", a=0, b=self.ir - self.dividend_yield),
            sigma=customFunc("cev", lamda=self.cev_lamda, p=self.cev_p),
            f_up=customFunc(
                "Exp Diff",
                isCall=True,
                k=self.strike,
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            ),
            f_dn=customFunc("constant", a=0),
            g=customFunc("RELU", a=-self.strike, b=1),
        )

    def tearDown(self):
        del self.mesh_BS_outright, self.mesh_Bache_outright
        logger.info("Tearing down class instrument...")

    def test_call(self):
        """Test the __call__ method of the mesh class"""
        # Firstly: compare Black model prices
        pde_price_1 = self.mesh_BS_outright(underlying=self.underlying)
        pde_price_1_ln = self.mesh_BS_log(underlying=np.log(self.underlying))
        bs_price_1 = undiscBSOptFwd(
            isCall=True,
            fwd=self.underlying,
            strike=self.strike,
            vol=self.bsvol,
            tau=self.tau,
        )
        logging.info(
            f"The % price difference between numerical (PDE) and analytical (BS) result is: {pde_price_1/bs_price_1-1}"
        )
        self.assertTrue(np.fabs(pde_price_1 / bs_price_1 - 1) < self.eps)
        self.assertTrue(np.fabs(pde_price_1_ln / bs_price_1 - 1) < self.eps)

        # Secondly: compare Black-Scholes model prices
        pde_price_2 = self.mesh_BS_outright_ir(underlying=self.underlying)
        pde_price_2_ln = self.mesh_BS_log_ir(underlying=np.log(self.underlying))
        bs_price_2 = BSOpt(
            isCall=True,
            spot=self.underlying,
            strike=self.strike,
            vol=self.bsvol,
            tau=self.tau,
            r=self.ir,
            q=self.dividend_yield,
        )
        logging.info(
            f"The % price difference between numerical (PDE) and analytical (BS) result is: {pde_price_2/bs_price_2-1}"
        )
        self.assertTrue(np.fabs(pde_price_2 / bs_price_2 - 1) < self.eps)
        self.assertTrue(np.fabs(pde_price_2_ln / bs_price_2 - 1) < self.eps)

        # Thirdly: compare Bachelier model prices
        pde_price_3 = self.mesh_Bache_outright(underlying=self.underlying)
        bach_price = undiscBachelierFwd(
            isCall=True,
            fwd=self.underlying,
            strike=self.strike,
            vol=self.bachvol,
            tau=self.tau,
        )
        logging.info(
            f"The % price difference between numerical (PDE) and analytical (Bachelier) result is: {pde_price_3/bach_price-1}"
        )
        self.assertTrue(np.fabs(pde_price_3 / bach_price - 1) < self.eps)

        # Fourth: compare CEV model prices
        pde_price_4 = self.mesh_CEV(underlying=self.underlying)
        cev_price = CEVSpot(
            isCall=True,
            spot=self.underlying,
            strike=self.strike,
            lamda=self.cev_lamda,
            p=self.cev_p,
            tau=self.tau,
            r=self.ir,
            q=self.dividend_yield,
        )
        logging.info(
            f"The % price difference between numerical (PDE) and analytical (CEV) result is: {pde_price_4/cev_price-1}"
        )
        self.assertTrue(np.fabs(pde_price_4 / cev_price - 1) < self.eps)


if __name__ == "__main__":
    unittest.main()
