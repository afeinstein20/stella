<p align="center">
  <img width = "500" src="./figures/stella_logo.png"/>
</p>

<p align="center">
  <a href="https://github.com/afeinstein20/stella/workflows/stella-tests/"><img src="https://github.com/afeinstein20/stella/workflows/stella-tests/badge.svg"/></a>
  <a href="https://arxiv.org/abs/2005.07710"><img src="https://img.shields.io/badge/read-the_paper-3C1370.svg?style=flat"/></a>
  <a href="https://afeinstein20.github.io/stella/"><img src="https://img.shields.io/badge/read-the_docs-3C1370.svg?style=flat"/></a>
  <a href="https://doi.org/10.21105/joss.02347">   <img src="https://joss.theoj.org/papers/10.21105/joss.02347/status.svg?color=D35968"></a>
</p>


</p>
stella is a Python package to create and train a neural network to identify stellar flares.
Within stella, users can simulate flares as a training set, run a neural network, and feed
in their own data to the neural network model. stella returns a probability at each data point
that that data point is part of a flare or not. stella can also characterize the flares identified.
</p>


To install stella with pip:

	pip install stella

Alternatively you can install the current development version of stella:

        git clone https://github.com/afeinstein20/stella
        cd stella
        python setup.py install

<p>
If your work uses the stella software, please cite <a href="https://ui.adsabs.harvard.edu/abs/2020JOSS....5.2347F/abstract">Feinstein, Montet, & Ansdell (2020)</a>.
</p>
<p>
If your work discusses the flare rate of young stars in the TESS Southern Hemisphere or the details of the CNNs, please cite <a href="https://ui.adsabs.harvard.edu/abs/2020arXiv200507710F/abstract">Feinstein et al. (AJ, accepted)</a>.
</p>

<p>
<b><u>Bug Reports, Questions, & Contributions</u></b>
</p>
<p>
stella is an open source project under the MIT license. 
The source code is available on GitHub. In case of any questions or problems, please contact us via the Git Issues. 
Pull requests are also welcome through the GitHub page.
</p>