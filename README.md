# XDGMM
This is a Python class for using Gaussian mixtures to do density estimation of noisy, heterogenous, and incomplete data using extreme deconvolution (XD) algorithms which is compatible with the scikit-learn machine learning methods. It implements both the [astroML](http://www.astroml.org/index.html) and [Bovy et al. (2011)](https://github.com/jobovy/extreme-deconvolution) algorithms, and extends the BaseEstimator class from [scikit-learn](http://scikit-learn.org/stable/) so that cross-validation methods will work. It allows the user to produce a conditioned model if values of some parameters are known.

[![Build Status](https://travis-ci.org/tholoien/XDGMM.svg?branch=master)](https://travis-ci.org/tholoien/XDGMM)
[![DOI](https://zenodo.org/badge/65572589.svg)](https://zenodo.org/badge/latestdoi/65572589)

## XDGMM Algorithms
The code currently supports the [astroML](http://www.astroml.org/index.html) and [Bovy et al. (2011)](https://github.com/jobovy/extreme-deconvolution) algorithms for XDGMM fitting and sampling. 

**Compatibility note:** Versions of astroML prior to May 20, 2015 contain a bug that causes an error with scoring samples. We recommend using later versions of astroML in order to avoid this.

## Machine Learning
This class is compatible with cross validation methods from [scikit-learn](http://scikit-learn.org/stable/). See the demo for an example of this functionality.

## Other Capabilities
XDGMM also allows the user to produce a conditional XDGMM distribution given values for some of the parameters used to create the model. For example, if parameters A, B, and C were used to fit a model and the value of C is known, you can produce a model for just parameters A and B that is conditioned on the known value of C.

## Contact

This is research in progress. All content is Copyright 2016 The Authors, and our code will be available for re-use under the MIT License (which basically means you can do anything you like with it but you can't blame us if it doesn't work). If you end up using any of the ideas or code in this repository in your own research, please cite [Holoien, Marshall, & Wechsler (2016)](http://adsabs.harvard.edu/abs/2016arXiv161100363H), and provide a link to this repo's URL: **https://github.com/tholoien/XDGMM**. However, long before you get to that point, we'd love it if you got in touch with us! You can write to us with comments or questions any time using [this repo's issues](https://github.com/tholoien/XDGMM/issues). We welcome new collaborators!

People working on this project:

* Tom Holoien (Ohio State, [@tholoien](https://github.com/tholoien/empiriciSN/issues/new?body=@tholoien))
* Phil Marshall (KIPAC, [@drphilmarshall](https://github.com/tholoien/empiriciSN/issues/new?body=@drphilmarshall))
* Risa Wechsler (KIPAC, [@rhw](https://github.com/tholoien/empiriciSN/issues/new?body=@rhw))
