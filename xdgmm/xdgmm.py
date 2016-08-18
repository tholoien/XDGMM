"""
Author: Tom Holoien
License: MIT

Extreme deconvolution solver

Implements both the astroML and Bovy et al. versions of XDGMM

Extends BaseEstimator from SciKit Learn to allow for CV training

Allows conditioning of the GMM based on a subset of the parameters.
"""

import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal

from sklearn.mixture import GMM as skl_GMM
from sklearn.base import BaseEstimator

from astroML.density_estimation import XDGMM as astroML_XDGMM
from astroML.utils import logsumexp

class XDGMM(BaseEstimator):
    """Extreme Deconvolution

    Class that can fit an extreme deconvolution (XD) model to the data

    Parameters
    ----------
    n_components: integer
        Number of gaussian components to fit to the data (default=1)
    n_iter: integer (optional)
        Number of EM iterations to perform (default=100).
        Not used with Bovy fitting method.
    tol: float (optional)
        Stopping criterion for EM iterations (default=1E-5).
    method: string (optional) 
        astroML or Bovy (default="astroML").
        
    Can be initialized with already known mu, alpha, and V:
    
    alpha: array_like (optional), shape = (n_components,)
        Weights for each gaussian (default=None).
    mu: array_like (optional), shape = (n_components, n_features)
        Means for each gaussian (default=None).
    V: array_like (optional), 
       shape  = (n_components, n_features, n_features)
       Covariance matrices for each gaussian (default=None).
    
    Can also be initialized from a file with a model saved in the format
        used by save_model and read_model. If a filename is given, the 
        model in the file will override any parameters passed to the 
        init function:
        
    filename: string (optional)
        Name of file from which to read in model parameters.

    Notes
    -----
    This implementation uses the astroML (http://www.astroml.org/) and
        Bovy et al. (arXiv 0905.2979) algorithms.
        
    This class extends the BaseEstimator class from scikit-learn and
        implements the necessary methods for cross-validation.
    """
    
    def __init__(self, n_components=1, n_iter=100, tol=1E-5,
                 method='astroML', random_state = None, V=None, mu=None,
                 weights=None,filename=None):
        
        if method != 'astroML' and method !='Bovy':
            raise ValueError("Fitting method must be 'astroML' or " +
                             "'Bovy'.")
        
        if filename is not None:
            self.read_model(filename)
        
        else:       
		    self.n_components = n_components
		    self.n_iter = n_iter
		    self.tol = tol
		    self.random_state = random_state
		    self.method=method
		    
		    # Model parameters. These are set by the fit() method but
		    # can be set at initialization.
		    self.V = V
		    self.mu = mu
		    self.weights = weights
		    
		    self.GMM=astroML_XDGMM(n_components,
		                           n_iter=n_iter,tol=tol,
		                           random_state=random_state)
		    self.GMM.mu=mu
		    self.GMM.V=V
		    self.GMM.alpha=weights

    def fit(self, X, Xerr):
        """Fit the XD model to data
        
        Whichever method is specified in self.method will be used.
        
        Results are saved in self.mu/V/weights and in the self.GMM
            object

        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data. 
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data.
        """
        
        if self.method=='astroML':
            self.GMM.n_components=self.n_components
            self.GMM.n_iter=self.n_iter
            self.GMM.fit(X, Xerr)
            
            self.V=self.GMM.V
            self.mu=self.GMM.mu
            self.weights=self.GMM.alpha
        
        if self.method=='Bovy':
            """
            Bovy extreme_deconvolution only imports if the method is
                'Bovy' (this is because installation is somewhat more
                complicated than astroML, and we don't want it to be
                required)
            
            As with the astroML method, initialize with a few steps of
                the scikit-learn GMM
            """
            from extreme_deconvolution import extreme_deconvolution\
                as bovyXD
            
            tmp_gmm = skl_GMM(self.n_components, n_iter=10,
                              covariance_type='full',
                              random_state=self.random_state)
            tmp_gmm.fit(X)
            self.mu = tmp_gmm.means_
            self.weights = tmp_gmm.weights_
            self.V = tmp_gmm.covars_
            
            logl=bovyXD(X,Xerr,self.weights,self.mu,self.V,
                        tol=self.tol)
            self.GMM.V=self.V
            self.GMM.mu=self.mu
            self.GMM.alpha=self.weights
            
        return self
    
    def score_samples(self, X, Xerr):
        """Return per-sample liklihood of the data under the model
        
        Uses the scikit-learn GMM.score_samples method to compute the 
            log probability of X under the model and return the 
            posterior probabilites of each mixture component for each 
            element of X
        
        Scores each data point in X separately so that each
            corresponding Xerr array can be folded into the covariance
            matrices and be included in the calculation (since the 
            scikit-learn GMM implementation does not include errors)

        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data.
                
        Returns
        -------
        logprob : array_like, shape = (n_samples, n_features)
            Log probabilities of each data point in X.
        
        responsibilities: array_like, shape = (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            data point in X.
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
        
        tmp_GMM=skl_GMM(self.n_components, n_iter=self.n_iter,
                        covariance_type='full',
                        random_state=self.random_state)
        tmp_GMM.weights_=self.weights
        tmp_GMM.means_=self.mu
        
        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        T = Xerr + self.V
        
        logprob=[]
        responsibilities=[]
        
        for i in range(X.shape[0]):
            tmp_GMM.covars_=T[i]
            lp,resp=tmp_GMM.score_samples(X[i])
            logprob.append(lp)
            responsibilities.append(resp)
        
        logprob=np.array(logprob)
        responsibilities=np.array(responsibilities)
        
        return logprob,responsibilities
    
    def predict(self, X, Xerr):
        """Predict a label for data
        
        Uses the results from score_samples to predict component
            memberships for each data point, as in the scikit-learn GMM

        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data.
                
        Returns
        -------
        C: array_like, shape = (n_samples,)
            Component memberships.
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")    
        
        logprob,responsibilities=self.score_samples(X,Xerr)
        return responsibilities.argmax(axis=1)
    
    def predict_proba(self, X, Xerr):
        """Predict posterior probability of data under each component
        in the model.
        
        Uses the results from score_samples to predict the probability
            for each data point under each component in the model, as in
            the scikit-learn GMM

        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data.
                
        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            The probability of the sample for each Gaussian
            component in the model.
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
        
        logprob,responsibilities=self.score_samples(X,Xerr)
        return responsibilities
        
    
    def logL(self, X, Xerr):
        """Compute the log-likelihood of data given the model
        
        Provides the log-likelihood of the data in X based on the 
            model in self.GMM, using the astroML XDGMM.logL method.
            (Regardless of which model was used to fit the data.)
            
        It is used to score the data for cross-validation.
        
        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data.
            
        Returns
        -------
        logL : float
            log-likelihood.
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
            
        return self.GMM.logL(X,Xerr)
        
    def score(self, X, Xerr):
        """Compute the score of data given the model
        
        Provides the log-likelihood of the data in X based on the 
            model as a score for scikit-learn cross-validation.
            
        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data.
            
        Returns
        -------
        score : float
            Score (log-likelihood).
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
        
        return self.logL(X, Xerr)
        
    def logprob_a(self, X, Xerr):
        """Compute the log probability of the data under the model
        
        Uses the astroML XDGMM.logprob_a to computer the log probability
            of the data under the model, regardless of which model was
            used to fit the data.

        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data. 

        Returns
        -------
        p: array_like, shape = (n_samples, n_components)
            Probabilities.
        """
    	if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
    	
    	return self.GMM.logprob_a(X,Xerr)

    def bic(self, X, Xerr):
        """Compute Bayesian information criterion for current model and
        proposed data.
        
        Computed in the same way as the scikit-learn GMM model computes
        the BIC.
        
        Parameters
        ----------
        X: array_like, shape = (n_samples, n_features)
            Input data.
        Xerr: array_like, shape = (n_samples, n_features, n_features)
            Error on input data. 

        Returns
        -------
        bic: float
            BIC for the model and data (lower is better).
        """
        logprob, _ = self.score_samples(X,Xerr)
        
        ndim = self.mu.shape[1]
        cov_params = self.n_components * ndim * (ndim + 1) / 2.
        mean_params = ndim * self.n_components
        n_params = int(cov_params + mean_params + self.n_components - 1)
        
        return (-2 * logprob.sum() + n_params * np.log(X.shape[0]))
    
    def sample(self, size=1, random_state=None):
        """Sample data from the GMM model
        
        This uses the astroML XDGMM.sample method, regardless of which
        method was used to fit the model.
        
        Parameters
        ----------
        size: int
            Number of samples to draw.
        random_state: random state
            Random state of the model (default = self.random_state)

        Returns
        -------
        sample : array_like, shape = (size, n_features)
            A sample of data points drawn from the model. 
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
        
        return self.GMM.sample(size,random_state)
    
    
    def condition(self, X, Xerr=None):
        """Condition the model based on known values for some
        features.
        
        Parameters
        ----------
        X : array_like, shape = (n_features,)
            An array of input values. Inputs set to NaN are not set, and 
            become features to the resulting distribution. Order is
            preserved.
        Xerr: array_like (optional), shape  = (n_features, )
            Errors for input values. Indeces not being used for 
            conditioning should be set to 0.0. If None, no additional
            error is included in the conditioning. (default=None).
            
        Returns
        -------
        cond_xdgmm: XDGMM object
            n_features = self.n_features-(n_features_conditioned)
            n_components = self.n_components
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
        
        new_mu=[]
        new_V=[]
        pk=[]
        
        not_set_idx=np.nonzero(np.isnan(X))[0]
        set_idx=np.nonzero(True-np.isnan(X))[0]
        x=X[set_idx]
        covars=self.V
        
        if Xerr is not None:
            for i in set_idx:
                covars[:,i,i] += Xerr[i]
        
        for i in range(self.n_components):
            a=[]
            a_ind=[]
            A=[]
            b=[]
            B=[]
            C=[]
            
            for j in range(len(self.mu[i])):
                if j in not_set_idx:
                    a.append(self.mu[i][j])
                else:
                    b.append(self.mu[i][j])
            
            for j in not_set_idx:
                tmp=[]
                for k in not_set_idx:
                    tmp.append(covars[i][j,k])
                A.append(np.array(tmp))
                
                tmp=[]
                for k in set_idx:
                    tmp.append(covars[i][j,k])
                C.append(np.array(tmp))
            
            for j in set_idx:
                tmp=[]
                for k in set_idx:
                    tmp.append(covars[i][j,k])
                B.append(np.array(tmp))
            
            a=np.array(a)
            b=np.array(b)
            A=np.array(A)
            B=np.array(B)
            C=np.array(C)
            
            mu_cond=a+np.dot(C,np.dot(np.linalg.inv(B),(x-b)))
            V_cond=A-np.dot(C,np.dot(np.linalg.inv(B),C.T))
            
            new_mu.append(mu_cond)
            new_V.append(V_cond)
            
            pk.append(multivariate_normal.pdf(x,mean=b,cov=B))
        
        new_mu=np.array(new_mu)
        new_V=np.array(new_V)
        pk=np.array(pk).flatten()
        new_weights=self.weights*pk
        new_weights=new_weights/np.sum(new_weights)
        
        return XDGMM(n_components=self.n_components, n_iter=self.n_iter, 
                     method=self.method, V=new_V,mu=new_mu, 
                     weights=new_weights)

    def save_model(self, filename='xdgmm.fit'):
        """Save the parameters of the model to a file
        
        Saves the model parameters to a file in a format that can be
            read by read_model.
        
        Parameters
        ----------
        filename: string
            Name of the file to save to. (default = 'xdgmm.fit')
        """
        if self.V is None or self.mu is None or self.weights is None:
            raise StandardError("Model parameters not set.")
        
        outfile=open(filename,'w')
        outfile.write('# XDGMM Model\n')
        outfile.write('# n_components  n_iter  tol  method   '    
                      +'random_state\n')
        outfile.write(str(self.n_components)+','+str(self.n_iter)
                      +','+str(self.tol)+','+self.method+','
                      +str(self.random_state)+'\n')
                      
        outfile.write('# weights\n')
        for i in range(len(self.weights)):
            outfile.write(str(self.weights[i]))
            if i != len(self.weights)-1: outfile.write(',')
            else: outfile.write('\n')
        
        outfile.write('# means\n')
        for i in range(len(self.mu)):
            for j in range(len(self.mu[i])):
                outfile.write(str(self.mu[i,j]))
                if j != len(self.mu[i])-1: outfile.write(',')
                else: outfile.write('\n')
        
        outfile.write('# covars\n')
        for i in range(len(self.V)):
            for j in range(len(self.V[i])):
                for k in range(len(self.V[i,j])):
                    outfile.write(str(self.V[i,j,k]))
                    if k != len(self.V[i,j])-1: outfile.write(',')
                    else: outfile.write('\n')
            if i != len(self.V)-1: outfile.write('#\n')
        
        outfile.close()
    
    def read_model(self,filename):
        """Read the parameters of the model from a file
        
        Read the parameters of a model from a file in the format saved
            by save_model and set the parameters of this model to those
            from the file
        
        Parameters
        ----------
        filename: string
            Name of the file to read from.
        """
        infile=open(filename,'r')
        inlines=infile.readlines()
        infile.close()
        
        params=inlines[2].split(',')
        self.n_components=int(params[0])
        self.n_iter=int(params[1])
        self.tol=float(params[2])
        self.method=params[3]
        if params[4]=='None\n': self.random_state=None
        else: self.random_state=int(params[4])
        
        weight_line=inlines[4].split(',')
        weights=[]
        for weight in weight_line:
            weights.append(float(weight))
        weights=np.array(weights)
        self.weights=np.array(weights)
        
        mu=[]
        for i in range(6,len(inlines)):
            if inlines[i]=='# covars\n':
                nextidx=i+1
                break
            
            tmp=[]
            line=inlines[i].split(',')
            for j in range(len(line)):
                tmp.append(float(line[j]))
            mu.append(np.array(tmp))
        mu=np.array(mu)
        self.mu=mu
        
        V=[]
        currV=[]
        i=nextidx
        while i<len(inlines):
            if inlines[i]=='#\n':
                V.append(np.array(currV))
                currV=[]
                i+=1
            
            line=inlines[i].split(',')
            tmp=[]
            for j in range(len(line)):
                tmp.append(float(line[j]))
            currV.append(np.array(tmp))
            i+=1
        V.append(np.array(currV))
        V=np.array(V)
        self.V=V
        
        self.GMM=astroML_XDGMM(n_components=self.n_components,
                               n_iter=self.n_iter,tol=self.tol,
                               random_state=self.random_state)
        self.GMM.mu=self.mu
        self.GMM.V=self.V
        self.GMM.alpha=self.weights