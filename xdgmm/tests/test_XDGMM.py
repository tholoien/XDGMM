"""
Test code for XDGMM class.
"""
import os
import unittest
import numpy as np
from sklearn.mixture import GaussianMixture as skl_GMM
from xdgmm import XDGMM

class XDGMMTestCase(unittest.TestCase):
    "TestCase class for XDGMM class."
    def setUp(self):
        """
        Set up each test with a new XDGMM object and some data.
        """
        self.xdgmm = XDGMM(n_components=3)
        self.files=[]
        
        """
        Use scikit-learn GaussianMixture for sampling some data points
        """
        self.gmm = skl_GMM(n_components=3, max_iter=10,
                           covariance_type='full',
                           random_state=None)
        self.gmm.weights_=np.array([0.3,0.5,0.2])
        self.gmm.means_=np.array([np.array([0,1]),np.array([5,4]),
                                  np.array([2,4])])
        self.gmm.covariances_=np.array([np.diag((2,1)),
                                        np.array([[1,0.2],[0.2,1]]),
                                        np.diag((0.3,0.5))])
        
        self.gmm.precisions_=np.linalg.inv(self.gmm.covariances_)
        self.gmm.precisions_cholesky_= np.linalg.cholesky(self.gmm.precisions_)
        
        self.X=self.gmm.sample(1000)[0]
        errs=0.2*np.random.random_sample((1000,2))
        self.Xerr = np.zeros(self.X.shape + self.X.shape[-1:])
        diag = np.arange(self.X.shape[-1])
        self.Xerr[:, diag, diag] = np.vstack([errs[:,0]**2, errs[:,1]**2]).T

    def tearDown(self):
        """
        Clean up files saved by tests
        """
        for fname in self.files:
        	os.remove('test.fit')

    def test_Fit(self):
        this_mu=self.xdgmm.mu
        this_V=self.xdgmm.V
        this_weights=self.xdgmm.weights

        self.xdgmm.fit(self.X, self.Xerr)

        self.assertIsNotNone(self.xdgmm.mu)
        self.assertIsNotNone(self.xdgmm.V)
        self.assertIsNotNone(self.xdgmm.weights)
    
    def test_Sample(self):
        self.xdgmm.fit(self.X, self.Xerr)
        sam=self.xdgmm.sample(1000)
        self.assertEqual(sam.shape,(1000,2))
    
    def test_Score(self):
        self.xdgmm.fit(self.X, self.Xerr)
        data=np.array([np.array([0,2]),np.array([4,4])])
        err=np.array([np.diag((0.2,0.1)),np.diag((0.15,0.15))])
        self.assertNotEqual(self.xdgmm.score(data,err),0)
    
    def test_ReadWrite(self):
        self.xdgmm.fit(self.X, self.Xerr)
        self.xdgmm.save_model('test.fit')
        xd2=XDGMM(filename='test.fit')
        
        self.assertLess(self.xdgmm.mu[0,0]-xd2.mu[0,0],1e-5)
        self.assertLess(self.xdgmm.V[0,0,0]-xd2.V[0,0,0],1e-5)
        self.assertLess(self.xdgmm.weights[0]-xd2.weights[0],1e-5)
        self.files.append('test.fit')
    
    def test_Condition(self):
        self.xdgmm.fit(self.X, self.Xerr)
        cond_xd=self.xdgmm.condition(X_input=np.array([np.nan,3.5]))
        
        self.assertEqual(cond_xd.mu.shape,(3,1))
        self.assertEqual(cond_xd.V.shape,(3,1,1))

if __name__ == '__main__':
    unittest.main()
