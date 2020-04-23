---
title: 'stella: Convolutional Neural Networks for Flare Identification in TESS'
tags:
  - Python
  - astronomy
  - PMS stars
  - stellar activity
  - stellar rotation
authors:
  - name: Adina D. Feinstein
    orcid: 0000-0002-9464-8101 
    affiliation: "1, 2"
  - name: Benjamin T. Montet
    orcid: 0000-0001-7516-8308
    affiliation: 3
  - name: Megan Ansdell
    affiliation: 4
  - name: Brian Nord
    affiliation: "5, 6"
  - name: Maximilian N. G{\"u}nther
    affiliation: "7, 8"
affiliations:
 - name: Department of Astronomy and Astrophysics, University of Chicago, 5640 S. Ellis Ave, Chicago, IL 60637, USA
   index: 1
 - name: NSF Graduate Research Fellow
   index: 2
 - name: School of Physics, University of New South Wales, Sydney, NSW 2052, Australia
   index: 3
 - name: Flatiron Institute, Simons Foundation, 162 Fifth Ave, New York, NY 10010, USA
   index: 4
 - name: Fermi National Accelerator Laboratory, P.O. Box 500, Batavia, IL 60510, USA
   index: 5
 - name: Kavli Institute for Cosmological Physics, University of Chicago, Chicago, IL 60637, USA
   index: 6
 - name: Department of Physics, and Kavli Institute for Astrophysics and Space Research, Massachusetts Institute of Technology, Cambridge, MA 02139, USA
   index: 7
 - name: Juan Carlos Torres Fellow
   index: 8

date: 4 May 2020
bibliography: paper.bib

aas-doi: 10.3847/xxxxx 
aas-journal: Astrophysical Journal
---

# Summary

Nearby young moving groups are kinematically bound systems of stars that are believed to have formed at the same time.
With all member stars having the same age, they provide snapshots of stellar and planetary evolution. 
In particular, young ($<$ 800 Myr) stars have increased levels of activity, seen in both fast rotation periods, large spot modulation, and increased flare rates.
Flare rates and energies can yield consequences for the early stages of planetary formation, particularly in the way of their atmospheres.
Models have demonstrated that the introduction of superflares ($> 5\%$ flux increase) are able to irreparably alter the chemistry of an atmosphere [@venot:2016] and expedite atmospheric photoevaporation [@lammer:2007]. 
Thus, understand flare rates and energies provide crucial keys for understanding the exoplanet population we see today.

Previous methods of flare detection have relied on detrending a light curve and using outlier detection heuristics for identifying flare events.
However, by relying on detrending first, low energy flares can easily be removed, where cadences within a flare can be flattened. 
Additionally, low energy flares likely fall below the sigma outlier threshold, biases the overall flare sample towards high energy flares.
As flares originate from the same physical processes and exhibit siilar time evolution (a sharp rise followed by an exopential decay), machine learning algorithms may prove suitable for identifying such features without light curve detrending.

`stella` is an open-source Python package for identifying flares in the *Transiting Exoplanet Survey Satellite* (*TESS*) two-minute data with convolutional neural networks (CNN).
Users have the option to build and train their own networks or the CNN models created for the work of @Feinstein:2020 are also included.
The training, validation, and test sets for our CNN used the flare catalog presented in @guenther:2019. These light curves are publically available through the Mikulski Archive for Space Telescopes and are not included in this package.
It takes approximately twenty minutes to create a CNN from scratch through the code and $<$ one minute to predict where flares are on a new light curve.
The package also allows users to measure rotation periods and fit flares to extract underlying flare parameters. Further documentation and tutorials can be found at \url{adina.feinste.in/stella}.

# Acknowledgements

We acknowledge contributions from Jacob Bean, Dorian Abbot, Thaddeus Komacek, James Owen, and Travis guru Rodrigo Luger.
This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. (DGE-1746045).
This work was funded in part through the NASA *TESS* Guest Investigator Program, as a part of Program G011237 (PI Montet).
