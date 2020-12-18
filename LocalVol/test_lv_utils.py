import unittest
import lv_utils
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Test_SpecialValues(unittest.TestCase):
    """Test the option prices are as expected for certain ad hoc parameter inputs. """

    def setUp(self):
        self.eps = 1e-10

    def test_fd1(self):
        self.assertEqual(lv_utils.fd1(1, 1, 1, 1), 0.5)

    def test_fd2(self):
        self.assertEqual(lv_utils.fd2(1, 1, 1, 1), -0.5)

    def test_d1(self):
        self.assertEqual(
            lv_utils.fd1(100, 100, 0.2, 10), lv_utils.d1(100, 100, 0.2, 10, 0, 0)
        )

    def test_d2(self):
        self.assertEqual(
            lv_utils.fd2(100, 100, 0.2, 10), lv_utils.d2(100, 100, 0.2, 10, 0, 0)
        )

    def test_d_Bach(self):
        self.assertEqual(lv_utils.d_Bache(1, 1, 1, 1), 0)
        self.assertEqual(lv_utils.d_Bache(1, 0, 1, 1), 1)

    def test_undiscBSOptFwd(self):
        self.assertTrue(
            np.fabs(lv_utils.undiscBSOptFwd(True, 1, 1, 1, 1) - 0.38292492254802624)
            < self.eps
        )

    def test_BlackImpVol(self):
        self.assertTrue(
            np.fabs(
                lv_utils.BlackImpVol(True, 1.0, 1.0, 1.0, 1.0, 0.38292492254802624)
                - 1.0
            )
            < self.eps
        )

    def test_undiscBachelierFwd(self):
        self.assertTrue(
            np.fabs(lv_utils.undiscBachelierFwd(True, 1, 1, 1, 1) - norm.pdf(0.0))
            < self.eps
        )


class Test_BoundaryCase(unittest.TestCase):
    """In CEV model, when p->1 it appproaches the Black model limit;
    when p->0 it approaches the Bachelier limit."""

    def test_undiscBSOptFwd(self):
        fwd, strike, vol, tau = 10, 1, 0.5, 1
        callPrice = lv_utils.undiscBSOptFwd(True, fwd, strike, vol, tau)
        self.assertTrue(np.fabs(callPrice / (fwd - strike) - 1) < 0.01)
        putPrice = lv_utils.undiscBSOptFwd(False, fwd, strike, vol, tau)
        self.assertTrue(np.fabs(putPrice < 0.01))


class Test_PriceEquivalence(unittest.TestCase):
    """Sometimes two pricers will yield identical results.
    This test checks if this is the case"""

    def setUp(self):
        self.eps = 1e-5

    def test_undiscCEVFwd(self):
        fwd, strike, lamda, tau, p = 1, 1, 1, 1, 0.99
        cevPrice = lv_utils.undiscCEVFwd(True, fwd, strike, lamda, tau, p)
        blackVolCEV = lv_utils.BlackImpVol(True, 1.0, fwd, strike, tau, cevPrice)
        blackPrice = lv_utils.undiscBSOptFwd(True, fwd, strike, lamda, tau)
        self.assertTrue(np.fabs(lamda / blackVolCEV - 1) < self.eps)
        self.assertTrue(np.fabs(cevPrice - blackPrice) < self.eps)

        fwd, strike, lamda, tau, p = 1, 1, 1, 1, 0
        cevPrice_2 = lv_utils.undiscCEVFwd(True, fwd, strike, lamda, tau, p)
        blackVolCEV_2 = lv_utils.BlackImpVol(True, 1.0, fwd, strike, tau, cevPrice_2)
        bachePrice_2 = lv_utils.undiscBachelierFwd(True, fwd, strike, lamda, tau)
        # self.assertTrue(np.fabs(lamda / blackVolCEV_2 - 1) < 1e-5)
        self.assertTrue(
            cevPrice_2 < bachePrice_2 and bachePrice_2 / cevPrice_2 - 1 < 4e-2
        )

    def test_CEVSpot(self):
        """Test the price equivalence of CEVSpot and undiscCEVFwd, under the scenario
        of zero interest rate and divident ratio"""
        spot, strike, lamda, tau, p = 1, 1, 1, 1, 0.5
        r, q = 0, 0
        callPrice1 = lv_utils.CEVSpot(True, spot, strike, lamda, tau, p, r, q)
        callPrice2 = lv_utils.undiscCEVFwd(True, spot, strike, lamda, tau, p)
        self.assertTrue(np.fabs(callPrice1 / callPrice2 - 1) < self.eps)

    def test_BSOpt(self):
        """Test the price equivalence between BSOpt and undiscBSOptFwd, under the scenario
        of zero interest rate and divident ratio"""
        spot, strike, vol, tau = 100, 100, 0.2, 10
        r, q = 0, 0
        callPrice1 = lv_utils.BSOpt(True, spot, strike, vol, tau, r, q)
        callPrice2 = lv_utils.undiscBSOptFwd(True, spot, strike, vol, tau)
        self.assertTrue(np.fabs(callPrice1 / callPrice2 - 1) < self.eps)

        putPrice1 = lv_utils.BSOpt(False, spot, strike, vol, tau, r, q)
        putPrice2 = lv_utils.undiscBSOptFwd(False, spot, strike, vol, tau)
        self.assertTrue(np.fabs(putPrice1 / putPrice2 - 1) < self.eps)


class Test_Monotonic(unittest.TestCase):
    """Test the monotonic change in option prices with certain underlying parameters"""

    def test_undiscBSOptFwd_monoVol(self):
        vols = np.linspace(1e-8, 1, 100)
        callPrices = lv_utils.undiscBSOptFwd(True, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = lv_utils.undiscBSOptFwd(False, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscBSOptFwd_monoTau(self):
        taus = np.linspace(0.04, 50, 100)
        callPrices = lv_utils.undiscBSOptFwd(True, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = lv_utils.undiscBSOptFwd(False, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscBachelierFwd_monoVol(self):
        vols = np.linspace(1e-8, 1, 100)
        callPrices = lv_utils.undiscBachelierFwd(True, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = lv_utils.undiscBachelierFwd(False, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscBachelierFwd_monoTau(self):
        taus = np.linspace(0.04, 50, 100)
        callPrices = lv_utils.undiscBSOptFwd(True, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = lv_utils.undiscBSOptFwd(False, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscCEVFwd_monoVol(self):
        vols = np.linspace(2e-1, 1, 100)
        callPrices = lv_utils.undiscCEVFwd(True, 100, 100, vols, 1, 0.5)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = lv_utils.undiscCEVFwd(False, 100, 100, vols, 10, 0.5)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscCEVFwd_monoTau(self):
        taus = np.linspace(0.04, 50, 100)
        callPrices = lv_utils.undiscCEVFwd(True, 1, 1, 0.2, taus, 0.5)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = lv_utils.undiscCEVFwd(False, 1, 1, 0.2, taus, 0.5)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_CEVSpot_monoStrike(self):
        spot, lamda, tau, p = 100, 2, 1, 0.8
        r, q = 0.03, 0.01
        strikes = np.linspace(80, 120, 100)
        callPrices = lv_utils.CEVSpot(True, spot, strikes, lamda, tau, p, r, q)
        self.assertTrue(np.all(np.diff(callPrices) < 0))
        putPrices = lv_utils.CEVSpot(False, spot, strikes, lamda, tau, p, r, q)
        self.assertTrue(np.all(np.diff(putPrices) > 0))


class Test_PutCallParity(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-10

    def test_BSOpt_pcparity(self):
        spot, vol, tau, p = 100, 0.3, 10, 1
        r, q = 0.03, 0.0
        df = np.exp(-r * tau)
        strikes = np.linspace(20, 200, 100)
        callPrices = lv_utils.BSOpt(True, spot, strikes, vol, tau, r, q)
        putPrices = lv_utils.BSOpt(False, spot, strikes, vol, tau, r, q)
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)

    def test_CEVSpot_pcparity(self):
        spot, lamda, tau, p = 100, 2, 10, 1  # Test boundary case for p=1
        r, q = 0.03, 0.01
        df = np.exp(-r * tau)
        strikes = np.linspace(60, 100, 100)
        callPrices = lv_utils.CEVSpot(True, spot, strikes, lamda, tau, p, r, q)
        putPrices = lv_utils.CEVSpot(False, spot, strikes, lamda, tau, p, r, q)
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)

        spot, lamda, tau, p = 100, 1, 10, 0.6  # Test case p<1
        r, q = 0.03, 0.01
        df = np.exp(-r * tau)
        strikes = np.linspace(30, 200, 1000)
        callPrices = lv_utils.CEVSpot(True, spot, strikes, lamda, tau, p, r, q)
        putPrices = lv_utils.CEVSpot(False, spot, strikes, lamda, tau, p, r, q)
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)

        spot, lamda, tau, p = 100, 2, 10, 0.5  # Test case p>1
        r, q = 0.03, 0.01
        df = np.exp(-r * tau)
        strikes = np.linspace(20, 100, 200)
        callPrices = lv_utils.CEVSpot(True, spot, strikes, lamda, tau, p, r, q)
        putPrices = lv_utils.CEVSpot(False, spot, strikes, lamda, tau, p, r, q)
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)


class Test_customFunc(unittest.TestCase):
    """Test the customerized functor and if they work as expected"""

    @classmethod
    def setUpClass(cls):
        logger.info("Setting up class object...")

    @classmethod
    def tearDownClass(cls):
        logger.info("Tearing down class object...")

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.eps = 1e-8
        self.func1 = lv_utils.customFunc("constant", a=1)
        self.func2 = lv_utils.customFunc("linear", a=1, b=10)
        self.func3 = lv_utils.customFunc("RELU", a=100, b=-200)
        self.func4 = lv_utils.customFunc("CEV", lamda=3.14, p=0.5)
        self.func5 = lv_utils.customFunc("EXP", c=3.14, d=0.5)
        self.func6 = lv_utils.customFunc("RELU", a=5)
        self.func7 = lv_utils.customFunc("Exp RELU", a=1.2, b=3.14, c=-24)
        self.func8 = lv_utils.customFunc("Exp RELU", a=1.2, b=3.14, c=-2400)
        self.func9 = lv_utils.customFunc(
            "EXP DIFF", isCall=True, k=107, r=0.05, q=0.03, T=7
        )

    def tearDown(self):
        logger.info("Tearing down class instrument...")

    def test_call(self):
        """Test the function call operator"""
        self.assertEqual(self.func1(1, 1), 1)
        self.assertEqual(self.func2(1, 5), 51)
        self.assertEqual(self.func3(0, 0), 100)
        self.assertEqual(self.func3(0, 0.51), 0)
        self.assertTrue(np.abs(self.func4(0, 7) - 8.307659116742816) < self.eps)
        self.assertTrue(np.abs(self.func5(0, 2.78) - 12.606629166401792) < self.eps)
        self.assertRaises(ValueError, self.func6, 1, 1)
        self.assertTrue(np.abs(self.func7(0, 2) - 616.5463965906735) < self.eps)
        self.assertEqual(self.func8(0, 2), 0)
        self.assertTrue(np.abs(self.func9(0, 105) - 9.709720226967306) < self.eps)


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
        self.mesh_BS_outright = lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=0,
            Mhigh=1000,
            interp="cubic spline",
            EquidistantGrid=True,
            m=400,
            n=300,
            r=lv_utils.customFunc("constant", a=0),
            mu=lv_utils.customFunc("constant", a=0),
            sigma=lv_utils.customFunc(
                "linear", a=0, b=self.bsvol
            ),  # b is BS volatility
            f_up=lv_utils.customFunc("RELU", a=-self.strike, b=1),
            f_dn=lv_utils.customFunc("constant", a=0),
            g=lv_utils.customFunc("RELU", a=-self.strike, b=1),
        )
        # Construct Black's model with x=ln(S)
        # dx(t) = dln(S)=dS(t)/S-1/2*dS^2/S^2
        #       = -1/2\sigma^2 dt+\sigma dW(t)
        self.mesh_BS_log = lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=np.log(self.underlying / 10),
            Mhigh=np.log(self.underlying * 10),
            interp="cubic spline",
            EquidistantGrid=True,
            m=300,
            n=300,
            r=lv_utils.customFunc("constant", a=0),
            mu=lv_utils.customFunc("constant", a=-0.5 * self.bsvol ** 2),
            sigma=lv_utils.customFunc("constant", a=self.bsvol),
            f_up=lv_utils.customFunc("Exp RELU", a=1, b=1, c=-self.strike),
            f_dn=lv_utils.customFunc("constant", a=0),
            g=lv_utils.customFunc("Exp RELU", a=1, b=1, c=-self.strike),
        )
        # Construct Black-Scholes model:
        # dS(t)=(r-a)S(t)dt + \sigma S(t)dW(t)
        self.mesh_BS_outright_ir = lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=0,
            Mhigh=500,
            interp="cubic spline",
            EquidistantGrid=True,
            m=300,
            n=300,
            r=lv_utils.customFunc("constant", a=self.ir),
            mu=lv_utils.customFunc("linear", a=0, b=self.ir - self.dividend_yield),
            sigma=lv_utils.customFunc(
                "linear", a=0, b=self.bsvol
            ),  # b is BS volatility
            f_up=lv_utils.customFunc(
                "Exp Diff",
                isCall=True,
                k=self.strike,
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            ),
            f_dn=lv_utils.customFunc("constant", a=0),
            g=lv_utils.customFunc("RELU", a=-self.strike, b=1),
        )
        # Construct Black-Scholes's model with x=ln(S)
        # dx(t) = dln(S)=dS(t)/S-1/2*dS^2/S^2
        #       = (r-a-1/2\sigma^2) dt+\sigma dW(t)
        self.mesh_BS_log_ir = lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=np.log(self.underlying / 10),
            Mhigh=np.log(self.underlying * 10),
            interp="cubic spline",
            EquidistantGrid=True,
            m=500,
            n=400,
            r=lv_utils.customFunc("constant", a=self.ir),
            mu=lv_utils.customFunc(
                "constant", a=self.ir - self.dividend_yield - 0.5 * self.bsvol ** 2
            ),
            sigma=lv_utils.customFunc("constant", a=self.bsvol),
            # f_up=lv_utils.customFunc("Exp RELU", a=1, b=1, c=-self.strike),
            f_up=lv_utils.customFunc(
                "Exp Diff 2",
                isCall=True,
                k=np.log(self.strike),
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            ),
            f_dn=lv_utils.customFunc("constant", a=0),
            g=lv_utils.customFunc("Exp RELU", a=1, b=1, c=-self.strike),
        )

        # Construct Bachelier model
        # dS(t) = \sigma dW(t)
        self.mesh_Bache_outright = lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=-300,
            Mhigh=500,
            interp="cubic spline",
            EquidistantGrid=True,
            m=600,
            n=400,
            r=lv_utils.customFunc("constant", a=0),
            mu=lv_utils.customFunc("constant", a=0),
            sigma=lv_utils.customFunc(
                "constant", a=self.bachvol
            ),  # a is Bachelier volatility
            f_up=lv_utils.customFunc("RELU", a=-self.strike, b=1),
            f_dn=lv_utils.customFunc("constant", a=0),
            g=lv_utils.customFunc("RELU", a=-self.strike, b=1),
        )
        # Construct CEV model
        # dS(t) = (r-a)S(t)dt+\lambda S^p dW(t)
        self.mesh_CEV = lv_utils.Mesh(
            self.tau,
            self.underlying,
            theta=self.theta,
            Mlow=self.underlying / 10,
            Mhigh=self.underlying * 5,
            interp="cubic spline",
            EquidistantGrid=True,
            m=500,
            n=200,
            r=lv_utils.customFunc("constant", a=self.ir),
            mu=lv_utils.customFunc("linear", a=0, b=self.ir - self.dividend_yield),
            sigma=lv_utils.customFunc("cev", lamda=self.cev_lamda, p=self.cev_p),
            f_up=lv_utils.customFunc(
                "Exp Diff",
                isCall=True,
                k=self.strike,
                r=self.ir,
                q=self.dividend_yield,
                T=self.tau,
            ),
            f_dn=lv_utils.customFunc("constant", a=0),
            g=lv_utils.customFunc("RELU", a=-self.strike, b=1),
        )

    def tearDown(self):
        del self.mesh_BS_outright, self.mesh_Bache_outright
        logger.info("Tearing down class instrument...")

    def test_call(self):
        """Test the __call__ method of the mesh class"""
        # Firstly: compare Black model prices
        pde_price_1 = self.mesh_BS_outright(underlying=self.underlying)
        pde_price_1_ln = self.mesh_BS_log(underlying=np.log(self.underlying))
        bs_price_1 = lv_utils.undiscBSOptFwd(
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
        bs_price_2 = lv_utils.BSOpt(
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
        bach_price = lv_utils.undiscBachelierFwd(
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
        cev_price = lv_utils.CEVSpot(
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