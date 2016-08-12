"""
    Test code for XDGMM class.
    """
from __future__ import absolute_import, division
import unittest

from xdgmm import XDGMM

class XDGMMTestCase(unittest.TestCase):
    "TestCase class for XDGMM class."
    def setUp(self):
        """
            Set up each test with a new XDGMM object.
            """
        self.xdgmm = XDGMM(n_components=1)

    def test_Fit(self):
        this_mu=self.xdgmm.mu
        this_V=self.xdgmm.V
        this_weights=self.xdgmm.weights

        # self.xdgmm.fit()

#        self.assertNotEqual(this_mu[0],self.xdgmm.mu[0])

if __name__ == '__main__':
    unittest.main()
