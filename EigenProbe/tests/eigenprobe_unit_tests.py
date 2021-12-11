import unittest
import time
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import EigenProbe.tip_modeling as TM
from common.baseclasses import AWA
warnings.simplefilter("ignore")

class TestEigenProbePML(unittest.TestCase):
    # Test that PML_profile and PML_domain produce expected shapes
    def test_PML_shape(self):
        Lx=400; Ly=200
        Nx=200; Ny=int(round(Nx*Ly/Lx)); Nfwhm=1; Nwidth=20; Nbuf=1;
        k=2;
        PML=TM.DefaultPML(size=(Lx,Ly),Nx=Nx,Nfwhm=1,Nwidth=Nwidth,k=k,Nbuf=Nbuf)
        self.assertTrue(PML.get_PML_profile().shape==(Nx,Ny))
        self.assertTrue(PML.get_PML_domain().shape==(Nx,Ny))

    # Test that DefaultPML correctly provides a buffer using the Nbuf parameter
    def test_PML_Nbuf(self):
        Lx=Ly=400;
        Nx=200; Nfwhm=1; Nwidth=20; Nbuf=1;
        k=2;
        dx=Lx/Nx
        PML=TM.DefaultPML(size=(Lx,Ly),Nx=Nx,Nfwhm=1,Nwidth=Nwidth,k=k,Nbuf=Nbuf)
        PML_slice=PML.get_PML_profile().cslice[0:,0]
        max=PML_slice.max()
        half_max=max/2
        x_max=PML_slice.axes[0][np.argmax(PML_slice)]
        self.assertTrue(round(x_max)==Lx/2-Nbuf*dx)

    # Test that DefaultPML correctly places the FWHM of the PML profile
    def test_PML_Nfwhm(self):
        Lx=Ly=400;
        Nx=200; Nfwhm=1; Nwidth=20; Nbuf=1;
        k=2;
        dx=Lx/Nx
        PML=TM.DefaultPML(size=(Lx,Ly),Nx=Nx,Nfwhm=1,Nwidth=Nwidth,k=k,Nbuf=Nbuf)
        PML_slice=PML.get_PML_profile().cslice[0:,0]
        max=PML_slice.max()
        half_max=max/2
        x_fwhm=PML_slice.axes[0][np.argmin(np.abs(PML_slice-half_max))]
        self.assertTrue(round(x_fwhm)==Lx/2-(Nbuf+Nfwhm)*dx)

    # Test that DefaultPML correctly cuts profile off
    def test_PML_Nwidth(self):
        Lx=Ly=400;
        Nx=200; Nfwhm=1; Nwidth=20; Nbuf=1;
        k=2;
        dx=Lx/Nx
        PML=TM.DefaultPML(size=(Lx,Ly),Nx=Nx,Nfwhm=1,Nwidth=Nwidth,k=k,Nbuf=Nbuf)
        PML_slice=PML.get_PML_profile().cslice[0:,0]
        xcutoff=0
        for x in PML_slice.axes[0][::-1]:
            if xcutoff==0 and PML.get_PML_domain().cslice[x,0]==0:
                xcutoff=x
        self.assertTrue(round(xcutoff)==Lx/2-Nwidth*dx)
        #self.assertTrue(round(x_fwhm)==Lx/2-(Nbuf+Nfwhm)*dx)

class TestEigenProbeSubstrates(unittest.TestCase):
    # Test that all eigenvalues in SubstrateDielectric are equal (within tolerance) to 1 by default
    def test_substrate_dielectric_eigenvalues(self):
        Dielectric=TM.SubstrateDielectric()
        self.assertTrue(np.all(Dielectric.eigenvalues-1<1e-5))

    # Test that SubstrateDielectric correctly sets beta
    def test_substrate_dielectric_beta(self):
        for beta in range(10):
            Dielectric=TM.SubstrateDielectric(beta=beta)
            self.assertTrue(np.all(Dielectric.eigenvalues-(1+beta)<1e-5))
