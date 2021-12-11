README:

Currently the lengths scales are set to each being equal to the tip radius of the SNOM. We can then translate that all into physical units.

Overall Outline of Functionality:

- Input Wavelength and Quality Factor of the Plasmons
    -This will define quantity sigma that controls the behavior of the eigenvalue/eigenvectors on the sample.

- Solve for the eigenfunctions and eigenvalues of the sample after using the real part from the value sigma.

- Correction Term Coloumb Kernel (Step Known as Massaging the Eigenvectors)

- Green's Function Method for Simulating the Interaction with a Bessel Function
    - Needs to define a term to take in the excitation energy
    - Mesh for scanning, currently set to 100x100, scaling up may cause us to run out of memory.
    

Outline of the Notebooks:

1. Creating Eigenfunctions
2. Inhomogenous Charge Distributions

Overall Organization of this Repository:

plasmon_modeling.py - Main module for simulating Plasmons

greens.py - Main Module for Constructing Green's Like Function and Futher Analysis

tools.py - May need an additional library for plotting tools and etc.

# Things to add / check
- Eigenfunctions and Eigenvalues to make sure that they are correct for specific examples (Unit Tests)

# Conda Environment
Conda environment is saved in the environment.yml

