# Particle clustering in turbulence: Prediction of spatial and statistical properties with deep learning
This is the code repository for: https://arxiv.org/abs/2210.02339

## Abstract
We investigate the utility of deep learning for modeling the clustering of particles that are aerodynamically coupled to turbulent fluids. Using a Lagrangian particle module within the Athena++ hydrodynamics code, we simulate the dynamics of particles in the Epstein drag regime within a periodic domain of isotropic forced hydrodynamic turbulence. This setup is an idealized model relevant to the collisional growth of micron to mm-sized dust particles in early stage planet formation. The simulation data is used to train a U-Net deep learning model to predict gridded three-dimensional representations of the particle density and velocity fields,  given as input the corresponding fluid fields. The trained model qualitatively captures the filamentary structure of clustered particles in a highly non-linear regime. We assess model fidelity by calculating metrics of the density field (the radial distribution function) and of the velocity field (the relative velocity and the relative radial velocity between particles). Although trained only on the spatial fields, the model predicts these statistical quantities with errors that are typically <10%. Our results suggest that, given appropriately expanded training data, deep learning could complement direct numerical simulations in predicting particle clustering within turbulent flows.

## File Layout
`model`  : Machine learning model

`params` : Model parameters of network

`plots`  : Code for plots 

`stats`  : Code for dust statistics 

`script` : Job scripts and athenainput 

`misc`   : Miscellaneous
