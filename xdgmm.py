"""
Author: Tom Holoien
License: MIT

Extreme deconvolution solver

Implements both the astroML and Bovy et al. versions of XDGMM (Bovy version not yet implemented)

Extends BaseEstimator from SciKit Learn to allow for CV training

Allows conditioning of the GMM based on a subset of the parameters.
"""

from time import time

import numpy as np
from scipy import linalg

from sklearn.mixture import GMM as skl_GMM
from sklearn.base import BaseEstimator

from astroML.density_estimation import XDGMM as astroML_XDGMM
from astroML.utils import logsumexp

class XDGMM(BaseEstimator):
	"""Extreme Deconvolution

    Fit an extreme deconvolution (XD) model to the data

    Parameters
    ----------
    n_components: integer
        number of gaussian components to fit to the data
    n_iter: integer (optional)
        number of EM iterations to perform (default=100)
    tol: float (optional)
        stopping criterion for EM iterations (default=1E-5)
    method: astroML or Bovy (default=astroML)
        
    Can be initialized with already known mu, alpha, and V
    V: Covariance matrices for each gaussian
    mu: Means for each gaussian
    alpha: Weights for each gaussian

    Notes
    -----
    This implementation follows Bovy et al. arXiv 0905.2979
    """
	def __init__(self, n_components, n_iter=100, tol=1E-5, method='astroML', V=None, mu=None, weights=None, verbose=False, random_state = None):         
		self.n_components = n_components
		self.n_iter = n_iter
		self.tol = tol
		self.verbose = verbose
		self.random_state = random_state
		self.method=method
		
		if method=='Bovy': raise NotImplementedError("Bovy fitting method is not yet implemented")

        # model parameters: these are set by the fit() method but can be set at initialization
		self.V = V
		self.mu = mu
		self.weights = weights
		
		#astroML XDGMM object that will be used for sampling and scoring
		self.GMM=astroML_XDGMM(self.n_components, n_iter=self.n_iter,tol=self.tol)

	def fit(self, X, Xerr, R=None):
		if self.method=='astroML':
			print self.n_components
			self.GMM.n_components=self.n_components
			
			self.GMM.fit(X, Xerr)
			
			self.V=self.GMM.V
			self.mu=self.GMM.mu
			self.weights=self.GMM.alpha
    	
		if self.method=='Bovy':
			raise NotImplementedError("Bovy fitting method is not yet implemented")
    		
		return self
    
	def logL(self, X, Xerr):
			'''
			gmm=astroML_XDGMM(self.n_components, n_iter=self.n_iter,tol=self.tol)
			
			gmm.V=self.V
			gmm.mu=self.mu
			gmm.alpha=self.weights
			'''
			return self.GMM.logL(X,Xerr)
    	
	def logprob_a(self, X, Xerr):
			'''
			gmm=astroML_XDGMM(self.n_components, n_iter=self.n_iter,tol=self.tol)
			
			gmm.V=self.V
			gmm.mu=self.mu
			gmm.alpha=self.weights
			'''
			return self.GMM.logprob_a(X,Xerr)
    
	def sample(self, size=1, random_state=None):
		if self.method=='astroML': return self.GMM.sample(size,random_state)
    
	def condition(self, X, indeces):
		"""
		Condition the model based on known values for some parameters.
		
		Parameters
		----------
		X : array_like, shape (n < n_features)
			List of data points with demension n < n_features.
		indeces: array_like, shape (X.shape)
			List of indeces in the GMM model that correspond to the values in X
		Returns
		-------
		XDGMM object with n_features = self.n_features-X.shape, n_components=self.n_components
		"""
		
		new_mu=[]
		new_V=[]
		
		#Note: Need to recalculate weights
		new_weights=self.weights
		
		for i in range(self.n_components):
			a=[]
			a_ind=[]
			A=[]
			b=[]
			B=[]
			C=[]
			
			for j in range(len(self.mu[i])):
				if j not in indeces:
					a.append(self.mu[i][j])
					a_ind.append(j)
				else:
					b.append(self.mu[i][j])
			
			for j in a_ind:
				tmp=[]
				for k in a_ind:
					tmp.append(self.V[i][j,k])
				A.append(np.array(tmp))
				
				tmp=[]
				for k in indeces:
					tmp.append(self.V[i][j,k])
				C.append(np.array(tmp))
			
			for j in indeces:
				tmp=[]
				for k in indeces:
					tmp.append(self.V[i][j,k])
				B.append(np.array(tmp))
			
			a=np.array(a)
			b=np.array(b)
			A=np.array(A)
			B=np.array(B)
			C=np.array(C)
			
			mu_cond=a+np.dot(C,np.dot(np.linalg.inv(B),(X-b)))
			V_cond=A-np.dot(C,np.dot(np.linalg.inv(B),C.T))
			
			new_mu.append(mu_cond)
			new_V.append(V_cond)
		
		new_mu=np.array(new_mu)
		new_V=np.array(new_V)
		
		return XDGMM(n_components=self.n_components,n_iter=self.n_iter,method=self.method,V=new_V,mu=new_mu,weights=new_weights)
	
	def score(self, X, Xerr):
		'''
		gmm=skl_GMM(n_components=self.n_components, n_iter=self.n_iter, covariance_type='full')
		gmm.covars_=self.V
		gmm.means_=self.mu
		gmm.weights_=self.weights
		
		return logsumexp(gmm.score(X))
		'''
		return self.logL(X, Xerr)
	'''
	def predict(self, X):
		gmm=skl_GMM(n_components=self.n_components, n_iter=self.n_iter, covariance_type='full')
		gmm.covars_=self.V
		gmm.means_=self.mu
		gmm.weights_=self.weights
		
		return gmm.predict(X)
	
	def predict_proba(self, X):
		gmm=skl_GMM(n_components=self.n_components, n_iter=self.n_iter, covariance_type='full')
		gmm.covars_=self.V
		gmm.means_=self.mu
		gmm.weights_=self.weights
		
		return gmm.predict_proba(X)
	'''