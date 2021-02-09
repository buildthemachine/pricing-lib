"""
This is the test file for Utils.vanilla_utils.py.
Author:     Yufei Shen
Date:       12/21/2020
"""

import unittest
import Utils.other_utils, Utils.vanilla_utils
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
        self.assertEqual(Utils.vanilla_utils.fd1(1, 1, 1, 1), 0.5)

    def test_fd2(self):
        self.assertEqual(Utils.vanilla_utils.fd2(1, 1, 1, 1), -0.5)

    def test_d1(self):
        self.assertEqual(
            Utils.vanilla_utils.fd1(100, 100, 0.2, 10),
            Utils.vanilla_utils.d1(100, 100, 0.2, 10, 0, 0),
        )

    def test_d2(self):
        self.assertEqual(
            Utils.vanilla_utils.fd2(100, 100, 0.2, 10),
            Utils.vanilla_utils.d2(100, 100, 0.2, 10, 0, 0),
        )

    def test_d_Bach(self):
        self.assertEqual(Utils.vanilla_utils.d_Bache(1, 1, 1, 1), 0)
        self.assertEqual(Utils.vanilla_utils.d_Bache(1, 0, 1, 1), 1)

    def test_undiscBSOptFwd(self):
        self.assertTrue(
            np.fabs(
                Utils.vanilla_utils.undiscBSOptFwd(True, 1, 1, 1, 1)
                - 0.38292492254802624
            )
            < self.eps
        )

    def test_BlackImpVol(self):
        self.assertTrue(
            np.fabs(
                Utils.vanilla_utils.BlackImpVol(
                    True, 1.0, 1.0, 1.0, 1.0, 0.38292492254802624
                )
                - 1.0
            )
            < self.eps
        )

    def test_undiscBachelierFwd(self):
        self.assertTrue(
            np.fabs(
                Utils.vanilla_utils.undiscBachelierFwd(True, 1, 1, 1, 1) - norm.pdf(0.0)
            )
            < self.eps
        )


class Test_BoundaryCase(unittest.TestCase):
    """In CEV model, when p->1 it appproaches the Black model limit;
    when p->0 it approaches the Bachelier limit."""

    def test_undiscBSOptFwd(self):
        fwd, strike, vol, tau = 10, 1, 0.5, 1
        callPrice = Utils.vanilla_utils.undiscBSOptFwd(True, fwd, strike, vol, tau)
        self.assertTrue(np.fabs(callPrice / (fwd - strike) - 1) < 0.01)
        putPrice = Utils.vanilla_utils.undiscBSOptFwd(False, fwd, strike, vol, tau)
        self.assertTrue(np.fabs(putPrice < 0.01))


class Test_PriceEquivalence(unittest.TestCase):
    """Sometimes two pricers will yield identical results.
    This test checks if this is the case"""

    def setUp(self):
        self.eps = 1e-5

    def test_undiscCEVFwd(self):
        fwd, strike, lamda, tau, p = 1, 1, 1, 1, 0.99
        cevPrice = Utils.vanilla_utils.undiscCEVFwd(True, fwd, strike, lamda, tau, p)
        blackVolCEV = Utils.vanilla_utils.BlackImpVol(
            True, 1.0, fwd, strike, tau, cevPrice
        )
        blackPrice = Utils.vanilla_utils.undiscBSOptFwd(True, fwd, strike, lamda, tau)
        self.assertTrue(np.fabs(lamda / blackVolCEV - 1) < self.eps)
        self.assertTrue(np.fabs(cevPrice - blackPrice) < self.eps)

        fwd, strike, lamda, tau, p = 1, 1, 1, 1, 0
        cevPrice_2 = Utils.vanilla_utils.undiscCEVFwd(True, fwd, strike, lamda, tau, p)
        blackVolCEV_2 = Utils.vanilla_utils.BlackImpVol(
            True, 1.0, fwd, strike, tau, cevPrice_2
        )
        bachePrice_2 = Utils.vanilla_utils.undiscBachelierFwd(
            True, fwd, strike, lamda, tau
        )
        self.assertTrue(np.fabs(lamda / blackVolCEV_2 - 1) < 3e-2)
        self.assertTrue(
            cevPrice_2 < bachePrice_2 and bachePrice_2 / cevPrice_2 - 1 < 4e-2
        )

    def test_CEVSpot(self):
        """Test the price equivalence of CEVSpot and undiscCEVFwd, under the scenario
        of zero interest rate and divident ratio"""
        spot, strike, lamda, tau, p = 1, 1, 1, 1, 0.5
        r, q = 0, 0
        callPrice1 = Utils.vanilla_utils.CEVSpot(
            True, spot, strike, lamda, tau, p, r, q
        )
        callPrice2 = Utils.vanilla_utils.undiscCEVFwd(True, spot, strike, lamda, tau, p)
        self.assertTrue(np.fabs(callPrice1 / callPrice2 - 1) < self.eps)

    def test_BSOpt(self):
        """Test the price equivalence between BSOpt and undiscBSOptFwd, under the scenario
        of zero interest rate and divident ratio"""
        spot, strike, vol, tau = 100, 100, 0.2, 10
        r, q = 0, 0
        callPrice1 = Utils.vanilla_utils.BSOpt(True, spot, strike, vol, tau, r, q)
        callPrice2 = Utils.vanilla_utils.undiscBSOptFwd(True, spot, strike, vol, tau)
        self.assertTrue(np.fabs(callPrice1 / callPrice2 - 1) < self.eps)

        putPrice1 = Utils.vanilla_utils.BSOpt(False, spot, strike, vol, tau, r, q)
        putPrice2 = Utils.vanilla_utils.undiscBSOptFwd(False, spot, strike, vol, tau)
        self.assertTrue(np.fabs(putPrice1 / putPrice2 - 1) < self.eps)


class Test_Monotonic(unittest.TestCase):
    """Test the monotonic change in option prices with certain underlying parameters"""

    def test_undiscBSOptFwd_monoVol(self):
        vols = np.linspace(1e-8, 1, 100)
        callPrices = Utils.vanilla_utils.undiscBSOptFwd(True, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = Utils.vanilla_utils.undiscBSOptFwd(False, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscBSOptFwd_monoTau(self):
        taus = np.linspace(0.04, 50, 100)
        callPrices = Utils.vanilla_utils.undiscBSOptFwd(True, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = Utils.vanilla_utils.undiscBSOptFwd(False, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscBachelierFwd_monoVol(self):
        vols = np.linspace(1e-8, 1, 100)
        callPrices = Utils.vanilla_utils.undiscBachelierFwd(True, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = Utils.vanilla_utils.undiscBachelierFwd(False, 1, 1, vols, 1)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscBachelierFwd_monoTau(self):
        taus = np.linspace(0.04, 50, 100)
        callPrices = Utils.vanilla_utils.undiscBSOptFwd(True, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = Utils.vanilla_utils.undiscBSOptFwd(False, 1, 1, 0.2, taus)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscCEVFwd_monoVol(self):
        vols = np.linspace(2e-1, 1, 100)
        callPrices = Utils.vanilla_utils.undiscCEVFwd(True, 100, 100, vols, 1, 0.5)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = Utils.vanilla_utils.undiscCEVFwd(False, 100, 100, vols, 10, 0.5)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_undiscCEVFwd_monoTau(self):
        taus = np.linspace(0.04, 50, 100)
        callPrices = Utils.vanilla_utils.undiscCEVFwd(True, 1, 1, 0.2, taus, 0.5)
        self.assertTrue(np.all(np.diff(callPrices) > 0))
        putPrices = Utils.vanilla_utils.undiscCEVFwd(False, 1, 1, 0.2, taus, 0.5)
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_CEVSpot_monoStrike(self):
        spot, lamda, tau, p = 100, 2, 1, 0.8
        r, q = 0.03, 0.01
        strikes = np.linspace(80, 120, 100)
        callPrices = Utils.vanilla_utils.CEVSpot(
            True, spot, strikes, lamda, tau, p, r, q
        )
        self.assertTrue(np.all(np.diff(callPrices) < 0))
        putPrices = Utils.vanilla_utils.CEVSpot(
            False, spot, strikes, lamda, tau, p, r, q
        )
        self.assertTrue(np.all(np.diff(putPrices) > 0))

    def test_sabr_monoVol(self):
        sig0 = np.linspace(0.02, 0.1, 10)
        prices = []
        for s in sig0:
            so = Utils.vanilla_utils.sabr_black_vol(
                isCall=True,
                x0=100,
                strike=100,
                tau=1,
                ir=0,
                dividend_yield=0,
                sabr_params=[s, 0.8, -0.2, 0.5],
            )
            prices.append(so.price())
        self.assertTrue(np.all(np.diff(prices) > 0))

    def test_sabr_monoAlpha(self):
        alphas = np.linspace(0, 2, 5)
        prices = []
        for alpha in alphas:
            so = Utils.vanilla_utils.sabr_black_vol(
                isCall=True,
                x0=100,
                strike=100,
                tau=1,
                ir=0,
                dividend_yield=0,
                sabr_params=[0.08, 0.8, -0.2, alpha],
            )
            prices.append(so.price())
        self.assertTrue(np.all(np.diff(prices) > 0))


class Test_PutCallParity(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-10

    def test_BSOpt_pcparity(self):
        spot, vol, tau = 100, 0.3, 10
        r, q = 0.03, 0.0
        df = np.exp(-r * tau)
        strikes = np.linspace(20, 200, 100)
        callPrices = Utils.vanilla_utils.BSOpt(True, spot, strikes, vol, tau, r, q)
        putPrices = Utils.vanilla_utils.BSOpt(False, spot, strikes, vol, tau, r, q)
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)

    def test_CEVSpot_pcparity(self):
        spot, lamda, tau, p = 100, 2, 10, 1  # Test boundary case for p=1
        r, q = 0.03, 0.01
        df = np.exp(-r * tau)
        strikes = np.linspace(60, 100, 1)
        callPrices = Utils.vanilla_utils.CEVSpot(
            True, spot, strikes, tau, lamda, p, r, q
        )
        putPrices = Utils.vanilla_utils.CEVSpot(
            False, spot, strikes, tau, lamda, p, r, q
        )
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)

        spot, lamda, tau, p = 100, 1, 10, 0.6  # Test case p<1
        r, q = 0.03, 0.01
        df = np.exp(-r * tau)
        strikes = np.linspace(30, 200, 1000)
        callPrices = Utils.vanilla_utils.CEVSpot(
            True, spot, strikes, tau, lamda, p, r, q
        )
        putPrices = Utils.vanilla_utils.CEVSpot(
            False, spot, strikes, tau, lamda, p, r, q
        )
        difference = spot * np.exp(-q * tau) - strikes * df
        self.assertTrue(np.max(np.fabs(callPrices - putPrices - difference)) < self.eps)

        spot, lamda, tau, p = 100, 2, 10, 0.5  # Test case p>1
        r, q = 0.03, 0.01
        df = np.exp(-r * tau)
        strikes = np.linspace(20, 100, 200)
        callPrices = Utils.vanilla_utils.CEVSpot(
            True, spot, strikes, tau, lamda, p, r, q
        )
        putPrices = Utils.vanilla_utils.CEVSpot(
            False, spot, strikes, tau, lamda, p, r, q
        )
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
        self.func1 = Utils.other_utils.customFunc("constant", a=1)
        self.func2 = Utils.other_utils.customFunc("linear", a=1, b=10)
        self.func3 = Utils.other_utils.customFunc("RELU", a=100, b=-200)
        self.func4 = Utils.other_utils.customFunc("CEV", lamda=3.14, p=0.5)
        self.func5 = Utils.other_utils.customFunc("EXP", c=3.14, d=0.5)
        self.func6 = Utils.other_utils.customFunc("RELU", a=5)
        self.func7 = Utils.other_utils.customFunc("Exp RELU", a=1.2, b=3.14, c=-24)
        self.func8 = Utils.other_utils.customFunc("Exp RELU", a=1.2, b=3.14, c=-2400)
        self.func9 = Utils.other_utils.customFunc(
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


class Test_funcWrapper(unittest.TestCase):
    """Tests if the pricing objects function as expected"""

    def setUp(self):
        logger.info("Setting up class instrument...")
        self.isCall = True
        self.spot = 100
        self.strike = 100
        self.tau = 1
        self.bo = Utils.vanilla_utils.Bachlier_obj(
            self.isCall,
            self.spot,
            self.strike,
            self.tau,
            vol=1,
            ir=0.05,
            dividend_yield=0.02,
        )
        self.co = Utils.vanilla_utils.CEV_obj(
            self.isCall,
            self.spot,
            self.strike,
            self.tau,
            lamda=0.6,
            p=0.8,
            ir=0.05,
            dividend_yield=0.02,
        )
        self.bso = Utils.vanilla_utils.GBM_obj(
            self.isCall,
            self.spot,
            self.strike,
            self.tau,
            vol=0.2,
            ir=0.05,
            dividend_yield=0.02,
        )
        self.so = Utils.vanilla_utils.sabr_black_vol(
            self.isCall,
            self.spot,
            self.strike,
            self.tau,
            ir=0,
            dividend_yield=0,
            sabr_params=[0.06, 0.8, -0.2, 0.5],
        )

    def test_price(self):
        """Test the price function of each class"""
        self.assertEqual(
            self.bo.price(),
            Utils.vanilla_utils.BachlierSpot(
                self.isCall, self.spot, self.strike, 1, 0.05, self.tau
            ),
        )
        self.assertEqual(
            self.co.price(),
            Utils.vanilla_utils.CEVSpot(
                self.isCall, self.spot, self.strike, self.tau, 0.6, 0.8, 0.05, 0.02
            ),
        )
        self.assertEqual(
            self.bso.price(),
            Utils.vanilla_utils.BSOpt(
                self.isCall, self.spot, self.strike, 0.2, self.tau, 0.05, 0.02
            ),
        )


if __name__ == "__main__":
    unittest.main()
