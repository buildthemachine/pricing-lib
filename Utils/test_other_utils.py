"""
This is the test file for other_utils.py.
Author:     Yufei Shen
Date:       2/22/2021
"""

import logging
import numpy as np
import unittest
import Utils.other_utils

from scipy.stats import norm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Test_norm_cdf(unittest.TestCase):
    """The norm cdf generation function from Graeme West paper"""

    def setUp(self):
        self.eps = 1e-7

    def test_norm_cdf(self):
        x = np.linspace(-20, 20, 100)
        target = norm.cdf(x)
        test = [Utils.other_utils.cnorma(s) for s in x]
        self.assertTrue(max(abs(test - target)) < self.eps)
