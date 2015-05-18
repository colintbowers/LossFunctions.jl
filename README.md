# LossFunctions

This is a Julia module for calculating various loss functions. The package is currently under development and should be done by early June.

I'm primarily building it to support another package I'm working on (ForecastEval). There is already a Julia package for loss (see Loss). So why build a new one? I have a fairly strong preference for building a new Type for each type of loss, as many of the loss functions I am interested in are parametrised. Also, I'm quite fond of how this kind of setup results in just one function name which then leverages multiple dispatch. As near as I can tell, the current Loss package instead relies on a large number of function names. I'm not advocating one approach as being better than the other - simply that I have a personal preference.

[![Build Status](https://travis-ci.org/colintbowers/LossFunctions.jl.svg?branch=master)](https://travis-ci.org/colintbowers/LossFunctions.jl)
