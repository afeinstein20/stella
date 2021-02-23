.. _overview:

About stella
=============

Summary of stella Functionality
--------------------------------

The purpose of stella is to identify flares in TESS short-cadence data with a convolutional neural network (CNN). 
In its simplest form, stella takes a pre-trained CNN (details provided in Feinstein et al. (submitted)) and a light curve (time, flux, and flux error) and returns a probability light curve. 
The cadences in the probability light curve are values between 0 and 1, where 1 means the CNN finds a flare there. 
Users also have the ability the train their own customized CNN architecture. The :ref:`quickstart tutorial <quickstart tutorial>` goes through these steps in more detail.

.. _Git Issue: http://github.com/afeinstein20/stella/issues
