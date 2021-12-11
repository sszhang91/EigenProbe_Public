# -*- coding: utf-8 -*-


"""

███████ ██  ██████  ███████ ███    ██ ██████  ██████   ██████  ██████  ███████ 
██      ██ ██       ██      ████   ██ ██   ██ ██   ██ ██    ██ ██   ██ ██      
█████   ██ ██   ███ █████   ██ ██  ██ ██████  ██████  ██    ██ ██████  █████   
██      ██ ██    ██ ██      ██  ██ ██ ██      ██   ██ ██    ██ ██   ██ ██      
███████ ██  ██████  ███████ ██   ████ ██      ██   ██  ██████  ██████  ███████

Synthesizing flatland nano-optics and photonics


@authors:
    Alex S. McLeod - alexsmcleod@gmail.com
    Michael Berkowitz - meb2235@columbia.edu
    William J.-C. Zheng - william.j.zheng@columbia.edu
    Leo F. B. Lo - cl3815@columbia.edu
"""

import os
import copy
import pickle
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from EigenProbe import utils
from EigenProbe.utils import Timer,progress
#from EigenProbe import Spectral as S
import Spectral as S
from NearFieldOptics.Materials.TransferMatrixMedia import LayeredMediaTM
from scipy import special as sp
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA

DEBUG = True

class CoulombOperatorFromLayers(S.GridIntegralOperator):
    """
        An implementation of GridIntegralOperator which represents the Coulomb
        kernel of a layered medium.

        Takes in a set of eigenfunctions as a basis and a layers object which
        will be used to construct the Coulomb kernel.
    """

    def __init__(self,basis,layers,freq=1e3,length_unit=10e-7,at_layer=2,**kwargs):
        assert isinstance(layers, LayeredMediaTM)
        assert isinstance(basis, S.Basis)

        def eval_kernel(qx,qy,layers,freq,length_unit,at_layer):
            Qy,Qx = np.meshgrid(qy,qx)
            Qs = np.sqrt(Qx**2+Qy**2)*1/length_unit
            omega = 2*np.pi*freq
            Kradial = layers.coulomb_kernel(freq, q=omega+Qs.ravel(),\
                                    layer_number=at_layer,mode="after")
            Kradial *= 1/length_unit
            return AWA(Kradial.reshape(Qs.shape),axes=[qx,qy],axis_names=['qx','qy']).squeeze()

        kff = lambda qx,qy: eval_kernel(qx,qy,layers,freq,length_unit,at_layer)
        x,y=basis.xy
        dx=(np.max(x)-np.min(x))/basis.size[0]
        self.convolver = numrec.QuickConvolver(kernel_function_fourier=kff,\
                                          shape=basis.shape,size=basis.size,\
                                           **kwargs)

        return super().__init__(lambda fs: [AWA(self.convolver(f),adopt_axes_from=basis.AWA) for f in fs],\
                                            dx=dx,order=0)

class CoulombOperatorUnscreened(S.GridIntegralOperator):
    """
        An implementation of GridIntegralOperator which represents an unscreened
        Coulomb kernel (i.e. 2*pi/q^2).

        Takes in a set of eigenfunctions as a basis.
    """

    def __init__(self,basis,**kwargs):
        assert isinstance(basis, S.Basis)
        self.convolver = numrec.QuickConvolver(kernel_function_fourier=utils.unscreened_coulomb_kernel_fourier,\
                                            shape=basis.shape,size=basis.size,**kwargs)
        x,y=basis.xy
        dx=(np.max(x)-np.min(x))/basis.size[0]
        return super().__init__(lambda fs: [AWA(self.convolver(f),adopt_axes_from=basis.AWA) for f in fs],\
                                            dx=dx,order=0)
            
class CoulombOperatorRetarded(S.GridIntegralOperator):
    """
        An implementation of GridIntegralOperator which represents an unscreened
        retarded Coulomb kernel of the first type (i.e. -2*pi*1j*kz/q^2).
        
        We don't yet know whether this one or the next one is formally correct.

        Takes in a set of eigenfunctions as a basis.
    """

    def __init__(self,basis,freq=1000,length_unit=20e-7,**kwargs):
        
        assert isinstance(basis, S.Basis)
        
        q0=2*np.pi*freq*length_unit
        kwargs.update(dict(q0=q0)) #These keyword arguments should be passed to the kernel function
        
        self.convolver = numrec.QuickConvolver(kernel_function_fourier=utils.unscreened_coulomb_kernel_retarded_fourier,\
                                            shape=basis.shape,size=basis.size, **kwargs)
        x,y=basis.xy
        dx=(np.max(x)-np.min(x))/basis.size[0]
        return super().__init__(lambda fs: [AWA(self.convolver(f),adopt_axes_from=basis.AWA) for f in fs],\
                                            dx=dx,order=0)
            
class CoulombOperatorRetarded2(S.GridIntegralOperator):
    """
        An implementation of GridIntegralOperator which represents an unscreened
        Coulomb kernel (i.e. -2*pi/(1j*kz)).
        
        We don't yet know whether this one or the previous one is formally correct.

        Takes in a set of eigenfunctions as a basis.
    """

    def __init__(self,basis=None,freq=1000,length_unit=20e-7,**kwargs):
        
        assert isinstance(basis, S.Basis)
        
        q0=2*np.pi*freq*length_unit
        kwargs.update(dict(q0=q0)) #These keyword arguments should be passed to the kernel function
        
        self.convolver = numrec.QuickConvolver(kernel_function_fourier=utils.unscreened_coulomb_kernel_retarded_fourier2,\
                                               shape=basis.shape,size=basis.size, **kwargs)
        x,y=basis.xy
        dx=(np.max(x)-np.min(x))/basis.size[0]
        return super().__init__(lambda fs: [AWA(self.convolver(f),adopt_axes_from=basis.AWA) for f in fs],\
                                            dx=dx,order=0)
            
class dZOperator(S.GridIntegralOperator):
    """
        An implementation of GridIntegralOperator which represents differentiation 
        in the out-of-plane direction of an (incoming) potential function.

        Takes in a set of eigenfunctions as a basis.
    """

    def __init__(self,basis=None,\
                 shape=None,size=None,
                 **kwargs):
        
        if basis is not None:
            assert isinstance(basis, S.Basis)
            shape=basis.shape
            size=basis.size
        else: assert shape is not None and size is not None
        
        #Propogation constant out-of-plane
        #@ASM 2020.10.06: TODO, maybe extend to finite-frequency, so fields near light line will matter more
        def kappa(kx,ky): return np.sqrt(kx**2+ky**2) #-k**2)
        
        self.convolver = numrec.QuickConvolver(kernel_function_fourier=kappa,\
                                            shape=shape,size=size,**kwargs)
        dx=size[0]/shape[0]
        return super().__init__(lambda fs: [AWA(self.convolver(f),adopt_axes_from=basis.AWA) for f in fs],\
                                            dx=dx,order=0)

class Translator(object):
    """
        Allows for the translation of the center point of functions within an xy mesh space.

        Works by evaluating a function at center of a much larger
        auxiliary xy mesh and then translating/truncating to the
        original mesh.  Nothing fancy, but good for performance.
    """

    def __init__(self,
                 xs=np.linspace(-1,1,101),
                 ys=np.linspace(-1,1,101),
                 f = lambda x,y: sp.jv(0,np.sqrt(x**2+y**2)),\
                 window=np.hamming,\
                 **kwargs):

        self.f=f
        self.window=window
        self.kwargs=kwargs

        self.xs,self.ys=xs,ys
        self.Nx,self.Ny=len(xs),len(ys)
        try: self.dx=np.abs(np.diff(xs)[0])
        except IndexError: self.dx=0
        try: self.dy=np.abs(np.diff(ys)[0])
        except IndexError: self.dy=0
        self.xmin=np.min(xs)
        self.ymin=np.min(ys)

        self.bigNx=2*self.Nx+1
        self.bigNy=2*self.Ny+1

        self.bigXs,self.bigYs=np.ogrid[-self.dx*self.Nx:+self.dx*self.Nx:self.bigNx*1j,
                                           -self.dy*self.Ny:+self.dy*self.Ny:self.bigNy*1j]
        
        self.bigF=self.f(self.bigXs,self.bigYs,**self.kwargs)

    def __call__(self,x0,y0):

        if self.dx: x0bar=(x0-self.xmin)/self.dx
        else: x0bar=0
        if self.dy: y0bar=(y0-self.ymin)/self.dy
        else: y0bar=0

        x0bar=int(x0bar)
        y0bar=int(y0bar)

        result=self.bigF[self.Nx-x0bar:2*self.Nx-x0bar,\
                         self.Ny-y0bar:2*self.Ny-y0bar]
            
        if self.window is not None:
            result=result*self.window(self.Nx)[:,np.newaxis]
            result=result*self.window(self.Ny)[np.newaxis,:]
            
        result=utils.normalize(result)

        return AWA(result,axes=[self.xs,self.ys])

class TranslatorPeriodic(object):
    """
        Allows for the translation of the center point of functions within an xy mesh space.

        Cell-periodic version of `Translator`.

        Works by evaluating a function at center of an
        auxiliary xy mesh and then translating/truncating to the
        original mesh.  Nothing fancy, but good for performance.
    """

    def __init__(self,
                 xs=np.linspace(-1,1,101),
                 ys=np.linspace(-1,1,101),
                 f = lambda x,y: sp.jv(0,np.sqrt(x**2+y**2)),\
                 window=np.hamming,\
                 **kwargs):

        self.f=f
        self.window=window
        self.kwargs=kwargs

        self.xs,self.ys=xs,ys
        self.Nx,self.Ny=len(xs),len(ys)
        try: self.dx=np.abs(np.diff(xs)[0])
        except IndexError: self.dx=0
        try: self.dy=np.abs(np.diff(ys)[0])
        except IndexError: self.dy=0
        self.xmin=np.min(xs)
        self.ymin=np.min(ys)

        if self.Nx % 2: #if odd
            self.Ndxleft=self.Ndxright=(self.Nx-1)/2
        else: #Zero coordinate will start second half
            self.Ndxleft=self.Nx/2; self.Ndxright=self.Ndxleft-1

        if self.Ny % 2: #if odd
            self.Ndyleft=self.Ndyright=(self.Ny-1)/2
        else: #Zero coordinate will start second half
            self.Ndyleft=self.Ny/2; self.Ndyright=self.Ndyleft-1

        #zero coordinate is now guaranteed in this "larger" mesh
        self.bigXs,self.bigYs=np.ogrid[-self.dx*self.Ndxleft:+self.dx*self.Ndxright:self.Nx*1j,
                                       -self.dy*self.Ndyleft:+self.dy*self.Ndyright:self.Ny*1j]

        self.bigF=self.f(self.bigXs,self.bigYs,**self.kwargs)
        
        #We can do windowing and normalizing in constructor, since the whole
        # present array will be used and simply rolled
        if self.window is not None:
            self.bigF*=self.window(self.Nx)[:,np.newaxis]
            self.bigF*=self.window(self.Ny)[np.newaxis,:]
            
        self.bigF=S.utils.normalize(self.bigF)

    def __call__(self,x0,y0):

        #`x0bar,y0bar` this is the index where we want the zero coordinate
        if self.dx: x0bar=(x0-self.xmin)/self.dx
        else: x0bar=0
        if self.dy: y0bar=(y0-self.ymin)/self.dy
        else: y0bar=0

        Nxshift=int(x0bar-self.Ndxleft)
        Nyshift=int(y0bar-self.Ndyleft)
        result=np.roll(self.bigF,Nxshift,axis=0)
        result=np.roll(result,Nyshift,axis=1)

        return AWA(result,axes=[self.xs,self.ys])

#--- Probe builders (using a standard format!)

class Eigenoscillator(object):
    """Provides a whole array of fake tip eigenoscillations from a single function
    `faketip_func`, that must have call signature `n,xs,ys,**kwargs`.  Constructed
    eigenoscillation functions will automatically scale xs,ys to tip size,
    and expand to xy grids so that faketip can safely be any combination of numpy ufuncs.
    Builder will pass static **kwargs down to the underlying function.

    This guy will also hold Rs and Ps as provided to constructor."""

    def __init__(self,eigenoscillation=utils.damped_bessel_eigenoscillation,\
                 tipsize=1,Ps=None,Rs=None,**kwargs):

        self.Rs=Rs
        self.Ps=Ps

        self.eigenoscillation=eigenoscillation
        self.tipsize=tipsize
        self.kwargs=kwargs

    def __call__(self,n):
        """This will build an eigenoscillation function at order `n`"""

        def wrapped_eigenoscillation(xs,ys):

            xs=np.asanyarray(xs)
            ys=np.asanyarray(ys)
            if xs.ndim==1 and ys.ndim==1:
                xs=xs[:,np.newaxis]; ys=ys[np.newaxis,:]

            xs/=self.tipsize; ys/=self.tipsize

            EO=self.eigenoscillation(n,xs,ys,**self.kwargs)
            #EO=utils.normalize(EO-np.mean(EO)) #Normalization at this stage is premature

            return EO

        return wrapped_eigenoscillation
    
probe_dir=os.path.join(os.path.dirname(__file__),'probe_library')

class LoadedEigenoscillator(Eigenoscillator): #We can just inherit `__call__` because I don't like copying code

    def __init__(self, fname,\
                 tipsize=1, ztip=0,amplitude=None,**kwargs):
        
        if os.path.exists(fname): path=fname
        else: path=os.path.join(probe_dir,fname)
            

        with open(path, 'rb') as handle:
            print('Loading probe from file "%s"...'%path)
            d = pickle.load(handle)

        self.d=d
        self.Ps = d['P']
        self.Rs = d['R']

        self.Ezs = d['Ez']
        self.Ers = d['Er']
        if amplitude is None:
            self.Vs = d['V'].interpolate_axis(ztip,axis=1,\
                                              extrapolate=True,bounds_error=False) #axis 1 is the ztip dimension
        else:
            self.Vs=np.mean(d['V'].cslice[:,0:amplitude],axis=1)

        self.max_n = self.Ezs.shape[0]

        self.tipsize=tipsize

        self.ns=self.Vs.axes[0]
        self.rs_norm=self.Vs.axes[1]

        #This step takes a while.
        #It could either be here in constructor, or passed down to runtime on `__call__` for
        # each eigenoscillation that is built.
        print('Building interpolators...')
        T=Timer()
        from scipy.interpolate import interp1d
        self.V_interpolators=[interp1d(self.rs_norm,self.Vs.cslice[n],\
                                       bounds_error=False,fill_value=self.Vs.cslice[n][-1],\
                                       **kwargs) \
                              for n in self.ns]
        print(T())

    # the function that will be returned
    def eigenoscillation(self,n,xs,ys,**kwargs): #TODO: enable access to other z-values?
        """
        This will build an eigenoscillation function at order `n`.

        Args:
            n: n is the order of the tip eigen-oscillation mode.
               n starts from 0.
            ztip: the distance between the apex of the tip and the sample.
            z: the height above the sample at which to evaluate the potential.

        Return: scalar potential at z=0, as a function of an x array and an y array

        Alex says: this function is super slow when fed to `Translator`s....
        """

        #check stuff
        #assert n<=self.max_n,'`n` must be no greater than %d'%self.max_n
        #V_n = self.Vs.cslice[n,ztip,:]

        interp=self.V_interpolators[n-1]

        ys,xs=np.meshgrid(ys,xs)
        rs=np.sqrt(xs**2+ys**2).ravel()
        Vs=interp(rs).reshape(xs.shape)
        
        return Vs

    def __call__(self,n):

        def wrapped_eigenoscillation(xs,ys):

            xs_norm=xs/self.tipsize; ys_norm=ys/self.tipsize

            EO=self.eigenoscillation(n,xs_norm,ys_norm)

            return EO

        return wrapped_eigenoscillation
    
class LoadedEigenoscillatorDemodulated(LoadedEigenoscillator): #We can just inherit `__call__` because I don't like copying code

    def __init__(self, fname,\
                 tipsize=1, harmonic=2, **kwargs):
        
        if os.path.exists(fname): path=fname
        else: path=os.path.join(probe_dir,fname)
            

        with open(path, 'rb') as handle:
            print('Loading probe from file "%s"...'%path)
            d = pickle.load(handle)

        self.Ps = d['P']
        self.Rs = d['R']

        self.Ezs = d['Ez']
        self.Ers = d['Er']
        self.Vs = d['V%i'%harmonic]

        self.max_n = self.Ezs.shape[0]

        self.tipsize=tipsize

        self.ns=self.Vs.axes[0]
        self.rs_norm=self.Vs.axes[1]

        #This step takes a while.
        #It could either be here in constructor, or passed down to runtime on `__call__` for
        # each eigenoscillation that is built.
        print('Building interpolators...')
        T=Timer()
        from scipy.interpolate import interp1d
        self.V_interpolators=[interp1d(self.rs_norm,self.Vs.cslice[n],\
                                       bounds_error=False,fill_value=self.Vs.cslice[n][-1],\
                                       **kwargs) \
                              for n in self.ns]
        print(T())
        
class EigenRasterer(object):
    
    def __init__(self,PS,\
                 excitation,collection=None,
                 normalize_collection=True,
                 normalize_excitation=True,
                 raster_collection=True,
                 raster_excitation=True,
                 recompute_eigenmodes=False,**kwargs):
        """
        This tool raster-scans an excitation field and/or a collection field
        over a Photonic2DSystem `PS`.
        
        By default, `collection=None` so that excitation and collection
        are the same field.  And if one is rastered, so is the other.
        
        The collection field vC will be normalized such that `(vC|Q|vC)=1`.
        If `normalize_excitation=True`, then similar normalization will be 
        applied to the excitation field vE such that `(vE|Q|vE)=1`.
        """
        
        #--- Retrieve eigenmodes from photonic system
        #assert isinstance(PS,Photonic2DSystem)
        self.PS=PS
        eigenmodes2D=PS.get_eigenmodes2D(recompute=recompute_eigenmodes)
        eigenmodesR=eigenmodes2D['eigenmode']
        eigenmodesL=eigenmodes2D['eigenmodeL']
        
        #--- Default collection
        if collection is None:
            collection=excitation
            raster_collection=raster_excitation
        assert excitation.shape==collection.shape,\
            "`excitation` and `collection` must be arrays of identical shape."
        
        #--- Build dZ operator
        dZ=dZOperator(size=utils.get_size(collection),\
                      shape=collection.shape).convolver
        Qcollection=dZ(collection)
        
        #--- Apply normalization to the collection field
        if normalize_collection:
            Ncol=np.sqrt(np.sum(collection*Qcollection)/(2*np.pi))
            Qcollection/=Ncol
        # and optionally to the excitation field (only useful if it's an eigenfield)
        if normalize_excitation:
            Qexcitation=dZ(excitation)
            Nexc=np.sqrt(np.sum(excitation*Qexcitation)/(2*np.pi))
            excitation/=Nexc

        #--- Define the actions of excitation and collection
        dx=np.diff(excitation.axes[0])[0]
        if raster_excitation:
            #Convolution kernel is a flipped-axes version of the probe we wish to raster
            exc_to_convolve=utils.invert_axes(excitation)
            E=numrec.QuickConvolver(kernel=exc_to_convolve/dx**2,**kwargs)
        else: E=lambda mode: np.sum(mode*excitation)
        if raster_collection:
            Qcol_to_convolve=utils.invert_axes(Qcollection)
            C=numrec.QuickConvolver(kernel=Qcol_to_convolve/dx**2,**kwargs)
        else: C=lambda mode: np.sum(mode*Qcollection)
        self.C=C
        self.E=E
        
        #--- Compute & store contributions from each eigenmode of `Photonic2DSystem`
        eigencontributions=[]
        print('Initializing eigenrasterer...')
        T=Timer()
        for psiR,psiL in zip(eigenmodesR,eigenmodesL):
            
            # We note that `R = eigvecsR @ np.diag(eigvals) @ eigvecsL.H`
            exc_el = E(np.conj(psiL)) # This is like `(psiL | exc)`
            coll_el = C(psiR) # This is like `(coll | Q | psiR)`
            # This is like `1/2pi (Qcoll|psiR) (psiL|exc)`
            eigencontributions.append(coll_el * exc_el / (2*np.pi))
            
        self.eigencontributions=np.array(eigencontributions)
        
        print(T())
            
    def __call__(self):
        
        #Instantly compute eigenreflectances from 2D system
        eigenreflectances=self.PS.get_eigenreflectances2D()
        
        #--- Sum over eigen-axis, with eigenreflectances the weighting factors
        Rs=eigenreflectances[:,np.newaxis,np.newaxis]
        beta=np.sum(Rs*self.eigencontributions,axis=0)
        
        # If we have a sized object, then size came from `eigencontributions`
        if isinstance(beta,np.ndarray) and beta.ndim:
            beta=AWA(beta,adopt_axes_from=self.PS.AWA)
            
        return beta

class EigenProbe(object):
    """
        Abstraction of a tip which can generate excitations

        Can output the tip eigenbases and the excitation functions
        
        TODO: Implement raster scanning on a sample.
    """

    def __init__(self,
                xs=np.linspace(-1,1,101), ys=np.linspace(-1,1,101),
                tipsize=1, Nmodes=5,
                eigenosc_fn=utils.damped_bessel_eigenoscillation,
                fname = None,
                periodic=False,\
                window=None,
                remove_mean=True,\
                **kwargs):

        self.tipsize=tipsize
        self.Nmodes=Nmodes
        self.remove_mean=remove_mean
        #self.N_tip_eigenbasis = N_tip_eigenbasis

        #Build functions for each eigenoscillation
        if not fname:
            self.eigenoscillator=Eigenoscillator(eigenosc_fn,tipsize=tipsize,**kwargs)
        else:
            self.eigenoscillator=LoadedEigenoscillator(fname,tipsize=tipsize,**kwargs) #kwargs could be ztip etc.

        #Transfer ownership of these to `EigenProbe`, or else we can just refer to eigenoscillator..
        self.Ps = self.eigenoscillator.Ps
        self.Rs = self.eigenoscillator.Rs
        self.eigenoscillations = [self.eigenoscillator(n) for n in np.arange(Nmodes)+1]

        #Build translators for each eigenoscillation
        if periodic: use_Translator=TranslatorPeriodic
        else: use_Translator=Translator
        self.translators = [use_Translator(xs,ys,eigenosc,window=window) \
                            for eigenosc in self.eigenoscillations]

    def __call__(self,x0,y0):
        tip_eb = [t(x0,y0) for t in self.translators]
        result=AWA(tip_eb, axes=[None,tip_eb[0].axes[0],tip_eb[0].axes[1]])
        
        #@EXPERIMENTAL: Remove mean from eigenoscillations
        if self.remove_mean:
            for eigenosc in result: eigenosc-=np.mean(eigenosc)
            
        return result
    
    def probe(self,photonic_system,X0,Y0,**kwargs):
        
        if not hasattr(X0,'__len__'): X0=[X0]
        if not hasattr(Y0,'__len__'): Y0=[Y0]
        X0=np.asarray(X0)
        Y0=np.asarray(Y0)
        assert len(X0)==len(Y0) and X0.ndim==1 and Y0.ndim==1
        
        excitations=np.array([self(x0,y0) for x0,y0 in zip(X0,Y0)])
        
        #unravel the "probe eigenindex" to a flat list of excitations
        multimode=False
        if excitations.ndim==4:
            multimode=True
            new_shape=(excitations.shape[0]*self.Nmodes,)+excitations.shape[2:]
            excitations.resize(new_shape)
            
        betas=photonic_system.get_reflection_coefficient(excitations,**kwargs)
        
        if not isinstance(betas,np.ndarray): return betas
        
        #re-ravel
        if multimode:
            old_size=(len(X0),self.Nmodes)
            betas=betas.reshape(old_size)
            betas=AWA(betas,axes=[None,np.arange(self.Nmodes)+1],\
                      axis_names=['position index','eigenfield index'])
        else:
            betas=AWA(betas,axes=[None],axis_names=['position index'])
            
        return betas
    
    def spectroscopy(self,photonic_system,X0,Y0,\
                     param,param_vals,**kwargs):
        #@TODO
        
        if not hasattr(X0,'__len__'): X0=[X0]
        if not hasattr(Y0,'__len__'): Y0=[Y0]
        X0=np.asarray(X0)
        Y0=np.asarray(Y0)
        assert len(X0)==len(Y0) and X0.ndim==1 and Y0.ndim==1
        
        excitations=np.array([self(x0,y0) for x0,y0 in zip(X0,Y0)])
        
        #unravel the "probe eigenindex" to a flat list of excitations
        multimode=False
        if excitations.ndim==4:
            multimode=True
            new_shape=(excitations.shape[0]*self.Nmodes,)+excitations.shape[2:]
            excitations.resize(new_shape)
            
        betas=photonic_system.get_reflection_coefficient(excitations,**kwargs)
        
        if not isinstance(betas,np.ndarray): return betas
        
        #re-ravel
        if multimode:
            old_size=(len(X0),self.Nmodes)
            betas=betas.reshape(old_size)
            betas=AWA(betas,axes=[None,np.arange(self.Nmodes)+1],\
                      axis_names=['position index','eigenfield index'])
        else:
            betas=AWA(betas,axes=[None],axis_names=['position index'])
            
        return betas
    
    def raster_probe(self,photonic_system,\
                     xlims=None,ylims=None,\
                     stride=1,chunks='auto',**kwargs):
        
        x,y=photonic_system.xy
        dx=np.diff(x)[0]
        if xlims is None: xlims=np.min(x),np.max(x)
        if ylims is None: ylims=np.min(y),np.max(y)
        
        X0,Y0=np.mgrid[xlims[0]:xlims[1]:dx*stride,
                       ylims[0]:ylims[1]:dx*stride]
        Nx,Ny=X0.shape
        beta_axes=[X0[:,0],Y0[0,:]]
        beta_axis_names=[r'$x_0$',r'$y_0$']
        
        X0=X0.flatten(); Y0=Y0.flatten()
        if chunks=='auto': chunks=Ny
        else: assert isinstance(chunks,int) and chunks>0
        chunksize=np.int(np.ceil(len(X0)/chunks))
        
        print('Commencing a %i x %i raster-scan, in %i chunks...'%(Nx,Ny,chunks))
        
        betas=[]
        for nchunk in range(chunks):
            X0_chunk=X0[nchunk*chunksize:(nchunk+1)*chunksize]
            Y0_chunk=Y0[nchunk*chunksize:(nchunk+1)*chunksize]
            betas_chunk=self.probe(photonic_system,X0_chunk,Y0_chunk,\
                                   **kwargs)
                
            if nchunk==0:
                kwargs['recompute']=False #assert that we don't need recalculation of matrices
                #determine beta shape, if `Nmodes>0` then it will be sized, else an empty tuple
                beta_shape=betas_chunk.shape[1:]
                
            betas+=list(betas_chunk)
            progress=(nchunk+1)/chunks*100
            print('Finished chunk %i of %i, %1.2f%% complete...'%(nchunk+1,chunks,progress))
            
        print('Probed %i positions.'%len(betas))
            
        #Reshape betas
        new_betas_shape=(Nx,Ny)+beta_shape
        betas=np.array(betas)
        betas.resize(new_betas_shape)
        
        if beta_shape:
            beta_axes+=[None]
            beta_axis_names+=['eigenfield index']
        
        return AWA(betas,axes=beta_axes,axis_names=beta_axis_names)
    
    def eigenraster(self,PS=None,normalize=True,return_eigenrasterers=False,\
                    **kwargs):
                
        #Get the probe field from `self`, and `QuickConvolver` needs to know the kernel positioned at `x,y=0,0`
        probe_fields=S.normalize(self(0,0))
        if probe_fields.ndim!=3: probe_fields=[probe_fields]
        
        betas=[]
        eigenrasterers=[]
        for i,probe_field in enumerate(probe_fields):
            print('Raster scanning eigenfield %i of %i...'%(i+1,len(probe_fields)))
            
            eigenrasterer=EigenRasterer(PS,\
                                         excitation=probe_field,collection=probe_field,
                                         normalize_excitation=True,
                                         raster_collection=True,
                                         raster_excitation=True,**kwargs)
            eigenrasterers.append(eigenrasterer)
            betas.append(eigenrasterer())
        
        psi=betas[0]
        betas=AWA(betas,axes=[None]+psi.axes,\
                  axis_names=['eigenfield index']+psi.axis_names)
        
        # Also return the `Eigenrasterer` objects if further rastering is needed
        if return_eigenrasterers: return betas.squeeze(),eigenrasterers
        else: return betas.squeeze()
    
class DipoleProbe(EigenProbe):
    
    def wrapped_dipole(self,n,xs,ys):
        
        field = utils.dipole_field(xs,ys,self.zfactor,\
                                   direction=self.direction)
            
        return field
    
    def __init__(self,xs=np.linspace(-1,1,101), ys=np.linspace(-1,1,101),
                 tipsize=1,direction=[0,1],pole=2,\
                 **kwargs):
        
        self.zfactor=5 #This corresponds to dipole z for a tip radius `a=1`
        self.direction=direction
        
        super().__init__(xs,ys,tipsize=tipsize,\
                         eigenosc_fn=self.wrapped_dipole,\
                         Ps=[pole],Rs=[1],Nmodes=1,**kwargs)
        
    def __call__(self,x0,y0):
        
        eigenfields=super().__call__(x0,y0)
        
        return eigenfields[0] #We only have one `eigenfield`, the one for the dipole!
    
class EigenProbeDemodulated(EigenProbe):

    def __init__(self,fname,
                xs=np.linspace(-1,1,101), ys=np.linspace(-1,1,101),
                tipsize=1, Nmodes=5,
                harmonic=2,\
                periodic=False,\
                window=None,
                remove_mean=True,\
                **kwargs):

        self.tipsize=tipsize
        self.Nmodes=Nmodes
        self.remove_mean=remove_mean
        #self.N_tip_eigenbasis = N_tip_eigenbasis

        #Build functions for each eigenoscillation
        self.eigenoscillator=LoadedEigenoscillatorDemodulated(fname,tipsize=tipsize,\
                                                              harmonic=harmonic,**kwargs) #kwargs could be ztip etc.

        #Transfer ownership of these to `EigenProbe`, or else we can just refer to eigenoscillator..
        self.Ps = self.eigenoscillator.Ps
        self.Rs = self.eigenoscillator.Rs
        self.eigenoscillations = [self.eigenoscillator(n) for n in np.arange(Nmodes)+1]

        #Build translators for each eigenoscillation
        if periodic: use_Translator=TranslatorPeriodic
        else: use_Translator=Translator
        self.translators = [use_Translator(xs,ys,eigenosc,window=window) \
                            for eigenosc in self.eigenoscillations]
    
    

#--- PML subclasses

class BasePML(object):  #For inheritance purposes

    def get_PML_domain(self): return self.PML_domain.copy()
    
    def get_PML_profile(self): return self.PML_profile.copy()

class DefaultPML(BasePML,S.GridLaplacian_aperiodic):
    
    def __init__(self,size,Nx,Nfwhm=4,Nwidth=40,k=2,Nbuf=1):
        
        #--- Introspect the simulation cell
        Lx,Ly=size
        X,Y=S.utils.get_XY(size,Nx)
        dx=np.diff(X[:,0])[0]
    
        #--- Build Perfectly matched layer (PML)
        # Actually, it's not perfectly matched at all, just a border of absorbing conductivity
        w_PML_frame = Nfwhm*dx #width of PML profile
        PML_frame1=1/(np.abs(X-Lx/2)**k+w_PML_frame**k)+1/(np.abs(X+Lx/2)**k+w_PML_frame**k)
        PML_frame2=1/(np.abs(Y-Ly/2)**k+w_PML_frame**k)+1/(np.abs(Y+Ly/2)**k+w_PML_frame**k)
        PML_profile=w_PML_frame**k*np.where(PML_frame1>PML_frame2,PML_frame1,PML_frame2)
        
        #--- set strictly equal to zero inside some box, then level the remaining PML
        w_PML=Nwidth*dx
        PML_domain=((np.abs(X)>(Lx/2-w_PML)) \
                   +(np.abs(Y)>(Ly/2-w_PML))) #a frame
        PML_profile*=PML_domain
        PML_profile[PML_domain]-=PML_profile[PML_domain].min()
        
        PML_profile/=np.mean(PML_profile[PML_domain])
            
        #--- Boundary extending all the way to edge has numerical issues
        buf=Nbuf*dx
        PML_profile*=((np.abs(X)<(Lx/2-buf))*(np.abs(Y)<(Ly/2-buf)))
        
        self.PML_domain=AWA(PML_domain,axes=[X.squeeze(),Y.squeeze()],axis_names=['x','y'])
        self.PML_profile=AWA(PML_profile,axes=[X.squeeze(),Y.squeeze()],axis_names=['x','y'])

        return super().__init__(dx,sigma = -1j*PML_profile) # `-1j` makes this an absorbing layer
    
class DefaultPMLX(BasePML,S.GridLaplacian_aperiodic):
    
    def __init__(self,size,Nx,Nfwhm=4,Nwidth=40,k=2,Nbuf=1):
        
        #--- Introspect the simulation cell
        Lx,Ly=size
        X,Y=S.utils.get_XY(size,Nx)
        dx=np.diff(X[:,0])[0]
    
        #--- Build Perfectly matched layer (PML)
        # Actually, it's not perfectly matched at all, just a border of absorbing conductivity
        w_PML_frame = Nfwhm*dx #width of PML profile
        PML_frame_x=1/(np.abs(X-Lx/2)**k+w_PML_frame**k)+1/(np.abs(X+Lx/2)**k+w_PML_frame**k)
        PML_profile=w_PML_frame**k*PML_frame_x+0*Y
        
        #--- set strictly equal to zero inside some box, then level the remaining PML
        w_PML=Nwidth*dx
        PML_domain=((np.abs(X)>(Lx/2-w_PML))+0*Y).astype(bool)
        PML_profile*=PML_domain
        PML_profile[PML_domain]-=PML_profile[PML_domain].min()
        
        PML_profile/=np.mean(PML_profile[PML_domain])
            
        #--- Boundary extending all the way to edge has numerical issues
        buf=Nbuf*dx
        PML_profile*=(np.abs(X)<(Lx/2-buf))
        
        self.PML_domain=AWA(PML_domain,axes=[X.squeeze(),Y.squeeze()],axis_names=['x','y'])
        self.PML_profile=AWA(PML_profile,axes=[X.squeeze(),Y.squeeze()],axis_names=['x','y'])

        return super().__init__(dx,sigma = -1j*PML_profile) # `-1j` makes this an absorbing layer
    
    
#--- Substrate subclasses

# Just use an alias here, we have nothing to add to the superclass yet
class BaseSubstrate(object): #For inheritance purposes

    def get_base_laplacian(self): return self.base_laplacian

class SubstrateDielectric(BaseSubstrate,S.SpectralOperator):
    
    def __init__(self,beta=0,base_laplacian=None,\
                 Lx=10,Nx=100,Ly=10,\
                 Nqmax=None,qys=None,include_const=False):
        
        #Take what we need from a `base_operator` of planewaves
        if base_laplacian is not None: assert isinstance(base_laplacian,S.SpectralLaplacian)
        else:
            base_laplacian=S.SpectralLaplacian_uniform(Lx=Lx,Nx=Nx,Ly=Ly,\
                                                      Nqmax=Nqmax,qys=qys,\
                                                      include_const=include_const)
        self.base_laplacian=base_laplacian
        
        planewaves=base_laplacian.eigenfunctions
        
        # This is a bit of a hack- we want 1 for each eigenfunction, but have to make unique keys
        betas=1+1e-8*np.arange(len(planewaves))
        
        eigpairs=dict(zip(betas,planewaves))
       # raise ValueError
        
        #No need to process eigenfunctions, `planewaves` are already orthonormal and AWA type
        super().__init__(eigpairs,process=False)
        
        self._inherit_basis(base_laplacian)
        
        self.Beta=S.utils.Constant(beta)
        
    def set_beta(self,beta): self.Beta.set_value(beta)
    
    def get_beta(self): return self.Beta.get_value()
    
class SubstrateFromLayers(BaseSubstrate,S.SpectralOperator):
    
    def __init__(self,layers,length_unit=10e-7,freq=1000,\
                 base_laplacian=None,\
                 Lx=10,Nx=100,Ly=10,\
                 Nqmax=1000,qys=None,include_const=False):
        
        assert hasattr(layers,'reflection_p')
        
        #Take what we need from a `base_operator` of planewaves
        if base_laplacian is not None: assert isinstance(base_laplacian,S.SpectralLaplacian)
        else:
            base_laplacian=S.SpectralLaplacian_uniform(Lx=Lx,Nx=Nx,Ly=Ly,\
                                                       Nqmax=Nqmax,qys=qys,\
                                                       include_const=include_const)
        self.base_laplacian=base_laplacian
                
        qs=np.sqrt(base_laplacian.eigenvalues)
        planewaves=base_laplacian.eigenfunctions
        
        # Compute rp from layers
        qs_wn=qs*1/length_unit
        rp=layers.reflection_p(freq,q=qs_wn)
        
        #Make sure all rp values are distinct
        rp[np.isnan(rp)]=0 # Prune potential nan values
        rp=utils.ensure_unique(rp,eps=1e-8)
            
        self.rp=AWA(rp,axes=[qs_wn],axis_names=[r'$q$ (cm$^{-1}$)'])
        
        #No need to process eigenfunctions, `planewaves` are already orthonormal and AWA type
        super().__init__((zip(rp,planewaves)),process=False)
        
        self._inherit_basis(base_laplacian)

#--- Photonic systems
    
class Photonic2DSystem(S.GeneralOperator):
    """
    Construct a photonic system of 2D materials `materials2D` on a `substrate`
    with (optionally) a perfectly matched layer `PML`.
    
    Specify substrate domain as a `SpectralLaplacian_uniform` instance, and
    2D materials `materials2D` as a sequence of `SpectralOperator` instances
    corresponding to eigenpairs of the Laplacian operator on their respective
    domains.
    
    The conductivity of 2D materials should be specified by `sigmas2D` in
    units of the conductivity specified by "native" plasmon wavelength `lambdap`.
    
    A composite basis of functions will be used to generate the response operator,
    constructed through QR decomposition of basis functions in `substrate` together
    with those exposed by `materials2D`; `Nbasis` restricts the maximum size of the
    composite basis.
    
    Generates the quasi-electrostatic response of a photonic system by:
        >>> responses = PhotonicSystem(excitations)
    and also generates generalized reflection coefficients by:
        >>> Rs = PhotonicSystem.get_reflection_coefficient(excitations)
        
    TODO:
        @ASM 2020.05.21:
        - Implement spectroscopy without recomputing excitation projections
        - Generalize to 2D materials encapsulated at interface ## within a stack
    """
    
    def __init__(self,substrate,materials2D,PML='default',V=None,\
                 beta_substrate=None,lambdap=1,sigmas2D=1,PML_amplitude=1,\
                 basis=None,Nbasis=2000,qmin=None,**kwargs):
        
        #--- Inspect the 2D materials
        types2D=(S.SpectralOperator,S.GridOperator)
        if isinstance(materials2D,types2D): materials2D=[materials2D] # expand single into list
        for material2D in materials2D:
            assert isinstance(material2D,types2D),\
                '2D materials must each be one of %s!'%str(types2D)
        if not hasattr(sigmas2D,'__len__'): sigmas2D=[sigmas2D]*len(materials2D)
        assert len(sigmas2D)==len(materials2D),\
            'If a sequence, `sigmas2D` should contain a value for each item of `materials2D`.'
        
        #--- Some introspection about simulation cell
        assert isinstance(substrate,BaseSubstrate)
        size=substrate.size
        Nx=substrate.shape[0]
        
        #--- Check PML
        if PML=='default':
            PML=DefaultPML(size,Nx,**kwargs)
        elif PML is None: PML=0
        else: assert isinstance(PML,BasePML)
        self.PML=PML
        
        # Build composite basis
        self.base_laplacian=None #No info on base laplacian yet
        if basis is not None: assert isinstance(basis,S.Basis)
        else:
            basis=self.build_basis(substrate,materials2D,Nbasis=Nbasis,qmin=qmin)
            self.base_laplacian=basis #Store basis to access base laplacian eigenvalues
        
        #--- Set the essential mutable amplitudes
        if hasattr(substrate,'Beta'): self.Beta_substrate=substrate.Beta #inherit substrate's handle
        else: self.Beta_substrate=S.utils.Constant(0) #else make a new mutable constant
        if isinstance(substrate,SubstrateFromLayers): beta_substrate=1 #Required to defer to substrate reflectivity
        if beta_substrate is not None: self.set_Beta_substrate(beta_substrate)
        
        self.PML_amplitude=S.utils.Constant(PML_amplitude)
        self.LambdaP=S.utils.Constant(lambdap)
        self.sigmas2D=[S.utils.Constant(sigma2D) for sigma2D in sigmas2D]
        
        # Set coulomb operator
        if V is None:
            V=CoulombOperatorUnscreened(basis) #`basis` is only ot provide length scales
        
        # Set operators list and feed to constructor
        operators=[substrate,PML,V]+list(materials2D) #Add operators in the correct order
        return super().__init__(operators,basis,\
                                operation=self._operation,\
                                process=False) #basis functions already normalized
    
    def build_basis(self,substrate,materials2D,Nbasis=2000,qmin=None):
        
        laplacian_sub=substrate.get_base_laplacian()
        add_materials2D=[]
        for i,material2D in enumerate(materials2D):
            
            # Don't do anything with `GridOperator` types, they'll just inherit basis
            if not isinstance(material2D,S.SpectralOperator): continue
        
            #This material will have nothing to add
            if material2D.shares_basis(laplacian_sub): continue
        
            add_materials2D.append(material2D)
        
        #If there's nothing to add to our basis, then we're done
        if not len(add_materials2D):
            self.msg('Defaulting to substrate basis functions...')
            return laplacian_sub
        
        #Only do QR decomposition etc. once on the whole set of laplacians
        self.msg('Constructing augmented basis with 2D materials...')
        basis=laplacian_sub.augmented_basis(*add_materials2D)
        
        #Add the laplacian matrices together and diagonalize
        self.msg('Building Laplacian operator in augmented basis...')
        T=Timer()
        laplacian_matrix=laplacian_sub.as_matrix_in(basis) \
                          + sum([mat2D.as_matrix_in(basis) \
                                 for mat2D in add_materials2D])
        laplacian=basis.expand_matrix(laplacian_matrix)
        self.msg(T())
        
        # Filter composite laplacian
        if qmin is None:
            L=np.max(laplacian.size)
            qmin=(2*np.pi/L) # Won't need plane waves much larger than the simulation size
        laplacian=laplacian.filtered_by_eigenvalues(lambda E: (E>=qmin**2))
        Nbasis=np.min((Nbasis, len(laplacian)-1))
        Emax=sorted(laplacian.eigenvalues)[Nbasis] # Max out the laplacian size
        laplacian=laplacian.filtered_by_eigenvalues(lambda E: (E<=Emax))
        laplacian[0]=np.zeros(laplacian.shape)+1 # Add in a constant function with eigenvalue 0
        self.msg('Size of composite laplacian: %i'%len(laplacian))
        
        return laplacian
    
    #--- Expose the essential mutable amplitudes
    def set_Beta_substrate(self,beta):
        """Set the reflection coefficient for the substrate; for layered substrates leave as `1`."""
        
        return self.Beta_substrate.set_value(beta)
    
    def set_PML_amplitude(self,PML_amplitude):
        """Set the amplitude of the PML; positive numbers indicate absorption."""
        
        return self.PML_amplitude.set_value(PML_amplitude)
    
    def set_LambdaP(self,lambdap):
        """Define the scale of 2D conductivity by providing a plasmon wavelength corresponding to `sigma2D=1`."""
        
        return self.LambdaP.set_value(lambdap)
    
    def set_Sigma2D(self,ind,sigma2D):
        """Set the conductivity for 2D material indexed by `ind`; generally as `sigma2D = sigma0*(1-1j/Q)` with
        Q the corresponding plasmon quality factor, and `sigma0` the overall scale of conductivity in units of
        that specified with `set_LambdaP`."""
        
        return self.sigmas2D[ind].set_value(sigma2D)
    
    def set_substrate(self,substrate,beta_substrate=None):
        
        assert isinstance(substrate,BaseSubstrate)
        
        if hasattr(substrate,'Beta'): self.Beta_substrate=substrate.Beta #inherit substrate's handle
        
        #Build new matrix for substrate
        P=substrate.project_into(self)
        M_substrate=P @ substrate.matrix @ P.H
        
        #Inject matrix into operands
        self._operator_matrices[0]=S.NormalMatrix(M_substrate)
       
        # Make sure layered reflection coefficient will be enabled
        if isinstance(substrate,SubstrateFromLayers): self.set_Beta_substrate(1)
        
    def _operation(self,substrate,PML,V,*materials2D):
        
        from builtins import sum
        
        #--- Definitions for 2D materials
        charge2D = sum([sigma2D*mat2D for sigma2D,mat2D in zip(self.sigmas2D,materials2D)])
        charge2D = charge2D + self.PML_amplitude*PML
        Qp = 2*np.pi/self.LambdaP #Plasmon momentum
        
        #--- Reflectivity of 2D materials
        # Order of 1/(...) doesn't matter because operators commute
        R2D=V/(2*np.pi)*charge2D/Qp*1/(V/(2*np.pi)*charge2D/Qp - 1)
        
        #--- Reflectivity of substrate
        Rsub=self.Beta_substrate * substrate #substrate reflectivity, equal to `1* rp(q)` for layered substrate
        
        #--- Compute screened responses
        R2D_scr  = 1/(1 - R2D*Rsub) * R2D * (1-Rsub)
        Rsub_scr = 1/(1 - Rsub*R2D) * Rsub * (1-R2D)
        
        return R2D_scr + Rsub_scr
    
    def get_Rmat(self,recompute=False):
        """Get the reflection matrix, cached for efficiency."""
        
        if not recompute:
            try: return self._Rmat
            except AttributeError: pass
        
        self.msg('Computing response matrix...')
        T=Timer()
        self._Rmat=self.get_matrix()
        self.msg(T())
        
        return self._Rmat
    
    def get_Qmat(self,recompute=False):
        """Get the Q matrix corresponding to operation `d/dz`, cached for efficiency."""
        
        if not recompute:
            try: return self._Qmat
            except AttributeError: pass
        
        dZ=dZOperator(basis=self)
        self._Qmat=dZ.as_matrix_in(self)
        
        return self._Qmat
        
    def get_reflection_coefficient(self,excitations,collections=None,\
                                   normalize_excitations=True,\
                                   normalize_collections=True,\
                                   recompute=True,\
                                   as_matrix=False):
        """Get generalized reflection coefficient in the 2D plane.
        
        Normalization only works as intended if the basis is complete.."""
        
        #--- Get vectors for excitations / collections
        if isinstance(excitations,np.matrix): vE = excitations
        else:
            self.msg("Projecting excitation vectors...")
            T=Timer()
            vE=self.vectors_from(excitations)
            self.msg(T())
        
        # Inherit collections as 
        if collections is None: vC=vE
        else:
            if isinstance(collections,np.matrix): vC = collections
            else:
                self.msg("Projecting collection vectors...")
                T=Timer()
                vC=self.vectors_from(collections)
                self.msg(T())
        
        #--- Get matrices, recomputing as requested
        Q=self.get_Qmat(recompute=False) #There will never be need to recompute `Qmat` unless basis changes
        R=self.get_Rmat(recompute=recompute)
        
        #--- Compute the response
        self.msg("Applying response matrix...")
        T=Timer()
        beta = np.array((vC.T @ Q @ R @ vE)/(2*np.pi)) #will be 2D
        
        #--- Normalize if desired
        if normalize_collections:
            Ncol=np.sqrt(np.array(vC.T @ Q @ vC)/(2*np.pi))
            Ncol=np.diag(Ncol).squeeze() #A 1d or 0d array
            #If multiple collections, make a column array
            if Ncol.ndim: Ncol=Ncol[:,np.newaxis]
            
        else: Ncol=1
        
        if normalize_excitations:
            Nexc=np.sqrt(np.array(vE.T @ Q @ vE)/(2*np.pi))
            Nexc=np.diag(Nexc).squeeze() #A 1d or 0d array
            #If multiple collections, make a row array
            if Nexc.ndim: Nexc=Nexc[np.newaxis,:]
            
        else: Nexc=1
        
        #--- Apply normalization, and coerce shape
        beta/=(Ncol*Nexc) #normalization either a number, or a 2d array
        
        if not as_matrix:
            #Take only diagonal elements
            beta=np.diag(beta).squeeze()
            #If just a number, dispense with array-ness
            if not beta.ndim: beta=np.complex(beta)
            
        self.msg(T())
            
        return beta
    
    # an alias
    R = get_reflection_coefficient
    
    #Computed the eigenmodes of 2D materials part of the operator
    def get_eigenmodes2D(self,recompute=False):
        """
        Compute eigenfunctions and eigenvalues of the screened Laplacian
        (V*L/2*pi) for combined 2D materials of the Photonic System.
        
        Eigenvalues (`qs`) will be in units of inverse characteristic length.
        Reflection intensity (`Rs`) of tese modes will also be computed as:
            `R_n = < phi_n | R | phi_n>`
        with `phi_n` the n'th eigenmode of the 2D materials and `R` is the
        PhotonicSystem reflectance.  Note that `phi_n` are approximately 
        eigenfunctions of R only with a dielectric substrate and PMLs disabled.
        
        Returns: dictionary of `qs`, `Rs` and `eigenmodes`, all as `AWA` 
                    instances indexed by real part of eigenvalues `qs` of
                    the screened Laplacian
        
        @ASM 2020.10.06 TODO: generalize to include PML?
        """
        
        if not recompute:
            try: return self.eigenmodes2D
            # If not available, proceed to compute
            except AttributeError: pass

        #operators order is `Substrate, PML, V, *materials2D`...
        #materials2D=copy.copy(self._operator_matrices[2:]) #This is V, followed by 2D materials
        #V=materials2D.pop(0) #First element is Coulomb Operator
        V=self._operator_matrices[2]
        materials2D=self._operator_matrices[3:]
        charge2D = sum([sigma2D*mat2D for sigma2D,mat2D in zip(self.sigmas2D,materials2D)])
        
        M=V/(2*np.pi)*charge2D

        if np.iscomplexobj(M): eig=np.linalg.eig; Mtype='non-Hermitian'
        else: eig=np.linalg.eigh; Mtype='Hermitian'

        self.msg("Diagonalizing new %s operator of size %s..."%(Mtype,str(M.shape)))
        T=Timer()
        qs,eigvecsR=eig(M)
        # We have a recomposition formula: `R = eigvecsR @ np.diag(eigvals) @ eigvecsR.I
        #                                     = eigvecsR @ np.diag(eigvals) @ eigvecsL.H
        # If right eigenvectors are orthonormal, then `.I` and `.H` are the same, and left/right eigenvectors are the same
        eigvecsL=eigvecsR.H.I
        self.msg(T())

        #expand the linear combinations of columns represented by eigenvectors
        self.msg("Expanding right and left eigenmodes...")
        eigenmodesR=self.expand_vectors(eigvecsR)
        eigenmodesL=self.expand_vectors(eigvecsL)
        self.msg(T())
        
        self.msg("Computing reflection intensity of 2D material eigenmodes...")
        pml_amp=self.PML_amplitude
        self.set_PML_amplitude(0)
        Rs=np.diag(eigvecsR.H @ self.matrix @ eigvecsR) #This is just a `<phi| R |phi>` structure
        self.set_PML_amplitude(pml_amp)
        
        #Now sort by q-value
        qs,Rs,eigenmodesR,eigenmodesL=zip(*sorted(zip(qs,Rs,eigenmodesR,eigenmodesL),\
                                         key=lambda trio: np.real(trio[0])))
        qs=np.array(qs)
        qs=AWA(qs,axes=[qs.real],axis_names=[r'Re$q$'])
        Rs=AWA(Rs,axes=[qs.real],axis_names=[r'Re$q$'])
        eigenmodesR=AWA(eigenmodesR,axes=[qs.real]+eigenmodesR[0].axes,\
                        axis_names=[r'Re$q$']+eigenmodesR[0].axis_names)
        eigenmodesL=AWA(eigenmodesL,axes=[qs.real]+eigenmodesL[0].axes,\
                        axis_names=[r'Re$q$']+eigenmodesL[0].axis_names)
        
        #@ASM 2020.10.05: TODO, perhaps abstractify this dictionary somehow
        #                 Like, return a new object, like `Photonic2DEigensystem`?
        self.eigenmodes2D=dict(q=qs,R=Rs,eigenmode=eigenmodesR,eigenmodeL=eigenmodesL)
            
        return self.eigenmodes2D
    
    # An alias for backwards compatibility
    get_materials2D_eigenmodes=get_eigenmodes2D
    
    def get_eigenreflectances2D(self,update=True):
        """
        The reflection is only physically for a uniform dielectric substrate with (local)
        reflectivity `beta`; the value given by `PS.Beta_substrate` is used to compute
        the reflectance.
        
        `update` will store the newly calculated reflectances in the `eigenmodes2D` dictionary.
        """
        
        #@ASM 2020.10.05: TODO, perhaps abstractify this dictionary update somehow
        eigenmodes2D=self.get_eigenmodes2D()
        q=eigenmodes2D['q']
        
        qp=2*np.pi/self.LambdaP.get_value()
        beta=self.Beta_substrate.get_value()
        
        N = beta - (1-beta) * q/qp
        D = 1 - (1-beta) * q/qp
        
        Rs = N/D
        
        if update: eigenmodes2D['R']=Rs
        
        return AWA(Rs,axes=[q.real],axis_names=[r'Re$q$'])
    
    get_reflection_coefficient_materials2D_eigenmode = get_eigenreflectances2D
    R2D = get_reflection_coefficient_materials2D_eigenmode #an alias
        
        
    