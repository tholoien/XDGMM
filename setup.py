from setuptools import setup

setup(
    name="xdgmm",
    version='1.0.9',
    author="Tom Holoien",
    author_email="tholoien@gmail.com",
    url="https://github.com/tholoien/XDGMM",
    packages=["xdgmm"],
    description="A wrapper class for the scikit-learn BaseEstimator class that implements both the astroML and Bovy et al. (2011) XDGMM methods.",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    use_2to3=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=["numpy", "astroML", "scikit-learn", "scipy"],
)
