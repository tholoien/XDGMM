# XDGMM
This is a Python class for using Gaussian mixtures to do density estimation of noisy, heterogenous, and incomplete data using extreme deconvolution (XD) algorithms which is compatible with the scikit-learn machine learning methods. It implements both the [astroML](http://www.astroml.org/index.html) and [Bovy et al. (2011)](https://github.com/jobovy/extreme-deconvolution) algorithms, and extends the BaseEstimator class from [scikit-learn](http://scikit-learn.org/stable/) so that cross-validation methods will work. It allows the user to produce a conditioned model if values of some parameters are known.

## XDGMM Algorithms
The code currently supports the [astroML](http://www.astroml.org/index.html) algorithm for XDGMM fitting and sampling. The [Bovy et al. (2011)](https://github.com/jobovy/extreme-deconvolution) algorithm will be implemented.

## Machine Learning
This class is compatible with cross validation methods from [scikit-learn](http://scikit-learn.org/stable/).

## Other Capabilities
XDGMM also allows the user to produce a conditional XDGMM distribution given values for some of the parameters used to create the model. For example, if parameters A, B, and C were used to fit a model and the value of C is known, you can produce a model for just parameters A and B that is conditioned on the known value of C.

## Contact

This is research in progress. All content is Copyright 2016 The Authors, and our code will be available for re-use under the MIT License (which basically means you can do anything you like with it but you can't blame us if it doesn't work). If you end up using any of the ideas or code in this repository in your own research, please cite **(Holoien et al, in preparation)**, and provide a link to this repo's URL: **https://github.com/tholoien/XDGMM**. However, long before you get to that point, we'd love it if you got in touch with us! You can write to us with comments or questions any time using [this repo's issues](https://github.com/tholoien/XDGMM/issues). We welcome new collaborators!

People working on this project:

* Tom Holoien (Ohio State, [@tholoien](https://github.com/tholoien/empiriciSN/issues/new?body=@tholoien))
* Phil Marshall (KIPAC, [@drphilmarshall](https://github.com/tholoien/empiriciSN/issues/new?body=@drphilmarshall))
* Risa Wechsler (KIPAC, [@rhw](https://github.com/tholoien/empiriciSN/issues/new?body=@rhw))
