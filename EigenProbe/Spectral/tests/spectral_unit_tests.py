import unittest
import time
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import Spectral as S

from inspect import isclass
from common.baseclasses import AWA
warnings.simplefilter("ignore")

class TestSpectralBasic(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        N=200;Nq=100;L=200;dx=L/N;
        self.SL = S.SpectralLaplacian_uniform(Lx=L,Ly=L,Nx=N,Nqmax=Nq)
        self.GL = S.GridLaplacian_periodic(dx)

    def test_eigval_scaling(self):
        SL2 = self.SL+(1*self.GL)
        SL5 = self.SL+(4*self.GL)
        SL10 = self.SL+(9*self.GL)

        self.assertEqual(round(np.mean(SL2.eigenvalues/self.SL.eigenvalues)),2)
        self.assertEqual(round(np.mean(SL5.eigenvalues/self.SL.eigenvalues)),5)
        self.assertEqual(round(np.mean(SL10.eigenvalues/self.SL.eigenvalues)),10)

    def test_inverse(self):
        Id = (2*self.SL)/(2*self.SL)

        self.assertAlmostEqual(round(np.mean(Id.eigenvalues)),1,places=7)

    def test_grid_vs_spectral(self):
        SL_tot = self.GL*(1/self.SL)
        mean_eigval = round(np.mean(SL_tot.eigenvalues))
        self.assertEqual(mean_eigval,1)

class TestSpectralAdvanced(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        N=400;L=500;Nq=100;
        Rx=0.85*L;Dx=(L-Rx)/2;

        self.SL = S.SpectralLaplacian_ribbon(Lx=L,Ly=L,Nx=N,Nqmax=Nq,Rx=Rx,x0=Dx)

    def test_eigenvalues(self):
        xs = np.arange(0,len(self.SL.eigenvalues),1)
        ys = 5.464e-5*(xs+1)
        mean_eigval_diff = np.mean(self.SL.eigenvalues-ys)

        self.assertTrue(mean_eigval_diff<1e-3)

    def test_normalized(self):
        result = round(float(np.sum(self.SL.eigenfunctions[0]**2)))
        self.assertEqual(result,1)

class TestSpectralLaplacians(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.SL_classes = []
        self.SLs = []
        self.kwargs = {
            "Lx": 10,
            "Ly": 20,
            "Rx": 5 ,
            "Ry": 10,
            "Nx": 100,
            "Nqmax": 500
        }
        self.kwargs["Ny"] = int((self.kwargs["Ly"]/self.kwargs["Lx"])*self.kwargs["Nx"])

        # Collect all classes in Spectral module which are
        # SpectralLaplacians and SpectralOperators
        for name,var in vars(S).items():
            if isclass(var):
                if issubclass(var,S.SpectralOperator) and issubclass(var,S.SpectralLaplacian) \
                    and (var is not S.SpectralLaplacian):
                    self.SL_classes.append(var)

        # Collect all classes which are "bounded"
        self.bounded_SL_classes = [x for x in self.SL_classes if "Rx" in S.utils.get_argnames(x)]

        # Create instances of all Spectral Laplacian classes
        for SL_cls in self.SL_classes:
            kwargs = {}
            cls_kwargs = S.utils.get_argnames(SL_cls)
            for key,val in self.kwargs.items():
                if key in cls_kwargs:
                    kwargs[key]=val
            self.SLs.append(SL_cls(**kwargs))

    def test_SpectralLaplacian_shapes(self):
        Nx = self.kwargs["Nx"]
        Ny = self.kwargs["Ny"]
        for SL in self.SLs:
            print("Testing shape of {}".format(SL.__class__))
            self.assertTrue(Nx==SL.shape[0])
            self.assertTrue(Ny==SL.shape[1])

    def test_SpectralLaplacian_sizes(self):
        Lx = self.kwargs["Lx"]
        Ly = self.kwargs["Ly"]
        for SL in self.SLs:
            print("Testing size of {}".format(SL.__class__))
            self.assertTrue(Lx==int(round(SL.size[0])))
            self.assertTrue(Ly==int(round(SL.size[1])))

    def test_SpectralLaplacian_bounded(self):
        Rx = self.kwargs["Rx"]
        Ry = self.kwargs["Ry"]
        Nx = self.kwargs["Nx"]
        Ny = self.kwargs["Ny"]
        for SL in self.SLs:
            for bounded_SL_class in self.bounded_SL_classes:
                if isinstance(SL,bounded_SL_class):
                    print("Testing boundedness of {}".format(SL.__class__))
                    evs = SL.eigenvalues
                    sums = []
                    if "Rx" in S.utils.get_argnames(bounded_SL_class):
                        sum_right = np.sum(SL[evs[-1]].cslice[Rx/2+Rx/Nx:,:])
                        sum_left  = np.sum(SL[evs[-1]].cslice[:-Rx/2-Rx/Nx,:])
                        sums.append(sum_right)
                        sums.append(sum_left)
                    if "Ry" in S.utils.get_argnames(bounded_SL_class):
                        sum_upper = np.sum(SL[evs[-1]].cslice[:,Ry/2+Ry/Ny:])
                        sum_lower = np.sum(SL[evs[-1]].cslice[:,:-Ry/2-Ry/Ny])
                        sums.append(sum_upper)
                        sums.append(sum_lower)
                    self.assertTrue(np.sum(sums)<1e-12)

    def test_SpectralLaplacian_orthonormal(self):
        for SL in self.SLs:
            if not isinstance(SL, S.SpectralLaplacian_disk):
                print("Testing orthonormality of {}".format(SL.__class__))
                U=S.utils.build_matrix(SL.eigenfunctions,SL.eigenfunctions)
                self.assertTrue(np.allclose(U,np.identity(U.shape[0]), atol=1e-1))

    def test_SpectralLaplacian_eigenfunctions(self):
        
        thresh=1e-8
        
        for SL in self.SLs:
            if not isinstance(SL, S.SpectralLaplacian_disk):
                print("Testing D^2u = q^2u of {}".format(SL.__class__))
                D = S.GridLaplacian_periodic(dx=SL.size[0]/SL.shape[0])
                eigvals = SL.eigenvalues
                eigfuncs = SL.eigenfunctions
                U=S.utils.build_matrix(D(eigfuncs),eigfuncs)
                mean_diff = np.mean(np.diag(U)-eigvals)
                renorm_U = np.diag(U)-mean_diff
                self.assertTrue(np.abs(np.sum(renorm_U-eigvals))<thresh)

class TestSpectralUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.C=S.utils.Constant(1)
        self.NM=S.utils.NormalMatrix([[1,1,0],[0,1,1],[1,0,1]])

    def test_timer(self):
        T = S.utils.Timer()
        time.sleep(1)
        msg = T()
        expected_msg = "\tTime elapsed: 1"
        self.assertEqual(msg[:len(expected_msg)],expected_msg)

    def test_inner_prod(self):
        psi = np.array([1+1j,2+2j])
        inner_prod_result = S.utils.inner_prod(psi,psi)
        self.assertEqual(inner_prod_result,10)

    def test_is_square(self):
        squ_mat = np.zeros((10,10))
        rec_mat = np.zeros((10,11))
        self.assertTrue(S.utils.is_square(squ_mat))
        self.assertFalse(S.utils.is_square(rec_mat))

    def test_is_normal(self):
        norm_mat=np.array([[1,1,0],[0,1,1],[1,0,1]])
        unnorm_mat=np.array([[1,1,0],[0,1,1],[1,0,0]])
        self.assertTrue(S.utils.is_normal(norm_mat))
        self.assertFalse(S.utils.is_normal(unnorm_mat))

    def test_Constant_set_get_value(self):
        val=2+2j
        self.C.set_value(val)
        self.assertEqual(self.C.get_value(),val)

    def test_Constant_Constant_add(self):
        val=1+1j
        self.C.set_value(val)
        self.assertEqual(self.C+self.C,val+val)

    def test_Constant_Constant_mul(self):
        val=2+2j
        self.C.set_value(val)
        self.assertEqual(self.C*self.C, val*val)

    def test_Constant_Constant_div(self):
        val=2+2j
        self.C.set_value(val)
        self.assertEqual(self.C/self.C,val/val)

    def test_Constant_num_add(self):
        val=2+3j
        self.C.set_value(val)
        self.assertEqual(self.C+val, val+val)

    def test_Constant_num_radd(self):
        val=2+3j
        self.C.set_value(val)
        self.assertEqual(val+self.C, val+val)

    def test_Constant_num_mul(self):
        val=2+3j
        self.C.set_value(val)
        self.assertEqual(self.C*val, val*val)

    def test_Constant_num_rmul(self):
        val=2+3j
        self.C.set_value(val)
        self.assertEqual(val*self.C, val*val)
