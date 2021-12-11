#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:37:07 2020

███████ ██████  ███████  ██████ ████████ ██████   █████  ██ 
██      ██   ██ ██      ██         ██    ██   ██ ██   ██ ██ 
███████ ██████  █████   ██         ██    ██████  ███████ ██ 
     ██ ██      ██      ██         ██    ██   ██ ██   ██ ██ 
███████ ██      ███████  ██████    ██    ██   ██ ██   ██ ███████ 

A toolset for nonlocal scattering solutions

@author: alexsmcleod@gmail.com
"""

import os
import copy
import numbers
import random
import builtins
import numpy as np
import pickle
from Spectral import utils
#Import into present namespace to support access in notebooks
from Spectral.utils import Timer,inner_prod,norm,normalize,Constant,NormalMatrix
from itertools import product
from common.baseclasses import ArrayWithAxes as AWA

#--- Cached projectors
cache_projectors=True
projectors_cache={}

#--- Fejer summation

def get_FejerMatrix(N):
    """Build a matrix to be applied to a vector before projection into some kind of Fourier basis.
    
    It will suppress high frequency components, but provides point-wise convergence to a desired
    function in the limit of an infinitely large basis."""
    
    FS=np.zeros((N,N))
    diag=2/N*np.arange(N,0,-1)
    np.fill_diagonal(FS,diag)
    
    return np.matrix(FS)

#--- `Basis` and subclasses

class Basis(dict):
    """
    A collection of functions indexed by integers, or more broadly
    by instances of hashable types (emulating a dictionary).

    Overrides `__setitem__` to normalize and cast input functions as AWA.
    """

    #--- Static & class methods

    @staticmethod
    def extract_columns(basis):

        if isinstance(basis,Basis): basis=basis.functions

        C=np.asarray(basis) #shares memory

        #This is the criterion that we have a list of 2D functions, unravel them
        if C.ndim>2:
            assert C.ndim==3
            npixels=np.prod(C.shape[1:]) #number of real-space pixels in these functions
            C=np.reshape(C,(len(C),npixels)).T #shares memory, #functions go down the columns now rather than rows

        return np.matrix(C,copy=False)

    debug=True

    @classmethod
    def msg(cls,msg):
        if cls.debug: print(msg)

    #--- Constructor

    def __init__(self,functions,size=(1,1),process=True,metric=None):
        """Functions can be a list or a dictionary where keys
        indicate function labels."""

        self._size=size
        self._process=process
        self._AWA=None
        
        #Set the metric
        if isinstance(functions,Basis): metric=functions.metric
        self.set_metric(metric)

        if isinstance(functions,dict):
            for key in functions: self[key]=functions[key]

            if isinstance(functions,Basis):
                #If we were actually a basis object, we have same basis, same metric!
                self._inherit_basis(functions)
        else:
            #If we did not get a list of tuples but a list of functions,
            # just assign ad hoc numeric labels to functions
            if (not isinstance(functions,zip)) and len(functions[0])!=2:
                functions=zip(range(len(functions)),functions)

            #Now we should have actual functions
            for key,val in functions: self[key]=val

        #Now that constructor is finished, turn processing back on by default
        self._process=True
        
    def __eq__(self, other):
        """This overload was especially necessary for functional
        comparisons like `self in [Basis1,Basis2,...] """
        
        if isinstance(other, type(self)): return self.shares_basis(other)
        else: return False

    #@ASM 2020.05.09: tested to work with copying `GeneralOperator`
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        #Also inherit entries from `self` using superclass `update`
        super(Basis,obj).update(self)
        return obj

    def __getstate__(self):

        d=self.__dict__.copy()
        d['self_dict']=dict(self).copy()

        return d

    def __setstate__(self,state):

        self_dict=state.pop('self_dict')
        dict.update(self,self_dict)
        self.__dict__.update(state)

    #--- Setting elements

    def _do_process(self):

        try: return self._process
        except AttributeError: return False

    def process_func(self,func):
        """Ensures normalization and casts to `AWA`"""

        try:
            #Ensure normalization
            func=utils.normalize(func,metric=self.metric)
            #Ensure correct axes
            return AWA(func,adopt_axes_from=self._getAWA())
        except ValueError:
            raise ValueError('`func` must have shape %s!'%self._getAWA().shape)

    def __setitem__(self, label, func):

        #before setting any items, we need an AWA
        if self._getAWA() is None: self._setAWA(func)

        if self._do_process():
            func=self.process_func(func)

        #If we have modified the basis, need a new hash
        self._update_basis()

        return super().__setitem__(label,func) #Feel free to overwrite previous labels
    
    def set_metric(self,metric):
        
        if metric is None: metric=1
        
        #We need a real-valued metric
        if isinstance(metric,numbers.Number):
            metric=np.float(metric)
        
        #Or we need real-valued array of correct shape
        else:
            assert isinstance(metric,np.ndarray)
            try: assert metric.shape is self.AWA.shape
            except AttributeError: pass #if `self._AWA` is not set
            metric=metric.astype(float) #We need float
        
        self._metric=metric

    #--- Managing metadata

    def _setAWA(self,func):

        assert func.ndim==2
        self._shape=func.shape

        if isinstance(func,AWA):

            self._size=[N*np.diff(ax)[0] for N,ax in zip(func.shape,\
                                                         func.axes)]
            self._AWA=func

        else:
            Nx=self._shape[0]
            X,Y=utils.get_XY(self._size,Nx,center=(0,0),sparse=True)
            axes=X.squeeze(),Y.squeeze()
            self._AWA=AWA(func,axes=axes,axis_names=['x','y'])

    def _getAWA(self):
        try: return copy.copy(self._AWA)
        except AttributeError: return None #It's not set yet!

    def _get_metric(self):
        
        if self._metric is None: return 1
        else: return copy.copy(self._metric)

    def _update_basis(self): self._hash=random.getrandbits(128)

    def _inherit_basis(self,other): self._hash=copy.copy(other._hash)

    def id_basis(self): return copy.copy(self._hash)

    def shares_basis(self,operator): return self.id_basis() is operator.id_basis()

    #--- Leveraging matrix representation

    def get_columns(self,basis=None):

        if basis is None: basis=self.functions

        return self.extract_columns(basis)
    
    def apply_metric(self,columns):
        
        #pass through if not explicit metric (implies `metric=1`)
        M=self.metric
        if M is None: return columns
        
        #Cast to column arrays to broadcast `M` over `columns`
        if isinstance(M,np.ndarray):
            M=self.extract_columns(M)
            M=np.asarray(M)
            columns=np.asarray(columns)
        
        return np.matrix(M*columns)

    def project_into(self,basis):
        """
        Build a projection transformation `P` for `self` projected into `basis`,
        specified by 1) a list of 2D array functions, 2) column vectors,
        or 3) a `Basis`.
    
        In bra-ket notation, `P` has ij'th element `<basis_i|self_j>`.
        
        If we have some matrix `M` in basis `self`, the transformation to a different
        basis will be given by `P @ M @ conj(P.T)` (assuming each basis is orthonormal
                                                    so that `P` is unitary).

        Meaningful if and only if `basis` is orthogonal.
        """
        
        #--- Try to load projector from cache
        if isinstance(basis,Basis) and cache_projectors:
            id_21=(basis.id_basis(),self.id_basis())
            try: return projectors_cache[id_21]
            except KeyError: pass

        columns1=self.get_columns(self)
        columns1=self.apply_metric(columns1) #should we have this operate in-place?
        columns2=self.get_columns(basis)

        shape=(columns2.shape[1],columns1.shape[1])
        self.msg('Building %ix%i inner product matrix...'%shape)

        T=utils.Timer()
        P21 = np.matrix(np.conj(columns2).T @ columns1)
        self.msg(T())
        
        #--- Try to store projector in cache
        if isinstance(basis,Basis) and cache_projectors:
            projectors_cache[id_21]=P21
            #Store inverse projection as well
            id_12=(self.id_basis(),basis.id_basis())
            projectors_cache[id_12]=P21.H

        return P21

    def vectors_from(self,functions):

        functions=np.asarray(functions)
        if functions.ndim<3: functions=[functions]
        F=self.extract_columns(functions)
        F=self.apply_metric(F)

        #Just do an inner product with each column in basis
        return np.matrix(np.conj(self.columns).T @ F)

    #An alias
    V = vectors_from

    def expand_vectors(self,vector,Fejer=None):
        """Expands a vector from our basis into a function.
        
        Could maybe be optimized to run fastor"""

        vector=np.asarray(vector)
        if not vector.ndim==2:
            vector=np.matrix(vector).T

        #This is the statement of assumption that a vector
        # represents linear combination of our eigenbasis
        F = self.columns @ vector

        functions=np.array([row.reshape(self.shape) for row in F.T])

        return AWA(functions,axes=[None]+self.AWA.axes,\
                   axis_names=[None]+self.AWA.axis_names).squeeze()
            
    def get_projection(self,functions,Fejer=False):
        """Project function(s) onto this basis."""
        
        v=self.vectors_from(functions)
        if Fejer:
            FM = get_FejerMatrix(len(v))
            v = FM @ v
        
        return self.expand_vectors(v)
    
    Fejer_resum = lambda functions: get_projection(functions,Fejer=True)

    #An alias
    F = expand_vectors

    def augmented_basis(self,*bases,smoothing=None,**kwargs):

        #If we got an empty list, return self
        if not len(bases): return self

        Cs=[basis.columns if isinstance(basis,Basis) \
            else basis for basis in bases]
        #Include self, otherwise this is hardly an instancemethod
        if self not in bases: Cs.append(self.columns)

        C=np.hstack(Cs) #combined columns of all vectors

        self.msg('Finding augmented basis by QR decomposition...')
        T=utils.Timer()

        #columns of C are a new basis
        Q,R=np.linalg.qr(C,**kwargs)

        tol=1e-9
        indep = np.abs(np.diag(R)) >  tol
        Q=Q[:,indep]
        self.msg('\tRemoved %i redundant vectors.'%(C.shape[1]-Q.shape[1]))

        self.msg(T())

        #Wrap into functions, smoothing if called for
        functions=[AWA(row.reshape(self.shape),\
                           adopt_axes_from=self._AWA) for row in Q.T]
        if smoothing:
            from common import numerical_recipes as numrec
            functions=[numrec.smooth(numrec.smooth(f,axis=0,window_len=smoothing),\
                                     axis=1,window_len=smoothing) for f in functions]

        #The column vectors & functions are automatically orthonormal
        return Basis(functions,size=self.size,process=False,metric=self.metric)

    def expand_matrix(self,M):
        """
        Generates a `SpectralOperator` description of (normal) matrix `M`
        using linear combinations of basis functions in `self`.
        """

        if not utils.is_normal(M):
            raise np.linalg.LinAlgError('Cannot construct `SpectralOperator` from a non-normal operator!')

        if np.iscomplexobj(M): eig=np.linalg.eig; Mtype='non-Hermitian'
        else: eig=np.linalg.eigh; Mtype='Hermitian'

        self.msg("Diagonalizing new %s operator of size %s..."%(Mtype,str(M.shape)))
        T=utils.Timer()
        eigvals,eigvecs=eig(M)

        #expand the linear combinations of columns represented by eigenvectors
        eigfuncs=self.expand_vectors(eigvecs,Fejer=False) #Straight expansion, otherwise not invertible

        # Don't process!  If one of `eigvecs` produces an
        # unnormalized eigenfunction, that's meaningful and
        # reflective of the new operator's output.  Anyway
        # `eigh` and `eig` already put all non-normalness
        # into eigenvalues, which eliminates our work.
        
        operator = SpectralOperator(zip(eigvals,eigfuncs),\
                                    process=False,metric=self.metric)
        self.msg(T())

        return operator

    #An alias
    M = expand_matrix

    def __getattribute__(self, name):

        if name=='shape': return self._getAWA().shape
        elif name=='size': return copy.copy(self._size)
        elif name=='functions': return list(self.values())
        elif name=='columns': return self.get_columns()
        elif name=='rank': return len(self)
        elif name=='AWA': return self._getAWA()
        elif name=='metric': return self._get_metric()
        elif name=='N': return np.prod(self._getAWA().shape)
        elif name=='xy': return self._getAWA().axes
        elif name=='XY': return self._getAWA().axis_grids

        return super().__getattribute__(name)

#--- `SpectralOperator` and subclasses

class SpectralOperator(Basis):
    """
    A spectral description of a normal operator, i.e. expressible
    by a set of (possibly complex) eigenvalues ond orthonormal
    eigenvectors:
    https://en.wikipedia.org/wiki/Normal_operator

    Must be constructed initially with an orthogonal set of
    functions; meanwhile `process=True` in the constructor will
    ensure that orthogonal functions are normalized.

    Possible extensions:

    I) `eigval_condition`: replaces `tol` to filter
    eigenvalues upon `__setitem__`
    """

    #--- Constructor

    def __init__(self, eigenpairs,\
                 size=(1,1),process=True,\
                 tol=None,metric=None):

        self.tol=tol

        super().__init__(eigenpairs,size=size,\
                         process=process,metric=metric)
            
    def __eq__(self, other):
        """This overload was especially necessary for functional
        comparisons like `self in [SpectralOperator1,SpectralOperator2,...] """
        
        #If basis is not shared or `other` is different type, we're done
        basis_eq=Basis.__eq__(self,other)
        if not basis_eq: return False
        
        #Now eigenvalues also have to be equal
        return np.allclose(self.eigenvalues,other.eigenvalues)

    #--- Setting elements

    def __setitem__(self, eigval,eigfunc):

        #Ensure we avoid exact degeneracy
        while eigval in self: eigval+=1e-12

        return super().__setitem__(eigval,eigfunc)

    def sort_by_eigenvalues(self):

        Es=self.eigenvalues
        fs=self.eigenfunctions

        idx=np.argsort(Es)
        newEs=Es[idx]
        newfs=[fs[i] for i in idx]

        #Unfortunately we have to disinherit our eigenbasis hash
        return SpectralOperator(zip(newEs,newfs),process=False)

    def filtered_by_eigenvalues(self,condition):

        Es=self.eigenvalues
        fs=self.eigenfunctions

        are_valid=condition(Es)
        newEs=Es[are_valid]
        newfs=[]
        for i,valid in enumerate(are_valid):
            if valid: newfs.append(fs[i])

        self.msg('Filtered %i eigenpairs.'%np.count_nonzero(are_valid==0))

        #Unfortunately we have to disinherit our eigenbasis hash
        return SpectralOperator(zip(newEs,newfs),\
                                process=False,tol=self.tol)

    #--- Leveraging matrix representation

    def __call__(self,functions):

        Vin=self.vectors_from(functions)

        Vout=self.matrix @ Vin

        return self.expand_vectors(Vout)

    #--- Manipulations between different bases

    def as_matrix_in(self,basis=None):
        """
        Build a matrix representation for `self` projected into `basis`,
        specified by 1) a list of 2D array functions, 2) column vectors,
        or 3) another `SpectralOperator`.

        Meaningful if and only if `basis` is orthogonal.
        """

        if basis is None \
            or (isinstance(basis,Basis) and self.shares_basis(basis)):
                return np.matrix(np.diag(self.eigenvalues))

        #This corresponds to `<self|basis>`
        # which we obtain as `<basis|self>.T`
        P=self.project_into(basis)

        return P @ self.matrix @ np.conj(P).T #projects from `basis` into ourselves and back out

    #An alias
    M=as_matrix_in
    
    def orthonormalize(self):
        """
        In the case that our underlying basis is not orthonormal, this
        method will use QR decomposition to identify a new basis and
        diagonalize the present operator in that basis.
        
        A resulting `SpectralOperator` is well defined, since any diagonal
        matrix in a non-orthogonal basis is still a normal operator, and
        has an orthogonal description (with modified eigenvalues).
        """
        
        basis=self.augmented_basis(self)
        Lmat=self.as_matrix_in(basis)
        
        return basis.expand_matrix(Lmat)
        

    def add(self,operator,augment=False,smoothing=None):
        """
        Add another `operator` to this one, with strength
        `strength`.  This is accomplished by ...

        Parameters
        ----------
        operator : `SpectralOperator` or dictionary of eigenpairs.
            Spectral operator to join, generally considered a perturbation
            on the parent operator, since the operator rank will not be
            enlarged.

        Returns
        -------
        joined : `SpectralOperator`
            Joined eigenbasis, generally larger in rank if
            `strength>0`, otherwise equal in rank to the
            parent eigenbasis.

        """

        if isinstance(operator,SpectralOperator) \
            and self.shares_basis(operator): # This may bypass the `augment` flag
                self.msg("Adding spectral operators in common basis...")
                eigvals = self.eigenvalues + operator.eigenvalues
                eigfuncs = self.eigenfunctions

                joined=SpectralOperator(zip(eigvals,eigfuncs),process=False)
                joined._inherit_basis(self) # Inherit hash label for basis

        else:
            if not augment:
                self.msg("Adding spectral operators in projected basis...")
                basis=self
                L1mat = self.matrix
                if isinstance(operator,np.matrix): L2mat=operator
                else: L2mat = operator.as_matrix_in(self) #We have performed change of basis from 2 to 1

            else:
                self.msg("Adding spectral operators in augmented basis...")
                assert isinstance(operator,SpectralOperator)
                basis=self.augmented_basis(self,operator)
                L1mat=self.as_matrix_in(basis) #a matrix
                L2mat=operator.as_matrix_in(basis)

            joined=basis.expand_matrix(L1mat + L2mat)

        return joined

    def mul(self,operator,direction='left',\
            augment=False,smoothing=None):
        """
        Multiply this operator onto another operator.
        This is accomplished by ...
        If the multiplication behaves as a projection, there is the possibility
        to produce a singular matrix.  Right now that will raise an error
        during diagonalization.

        Parameters
        ----------
        operator : `SpectralOperator` or dictionary of eigenpairs.
            Spectral operator to join, generally considered a perturbation
            on the parent operator, since the operator rank will not be
            enlarged.

        Returns
        -------
        joined : `SpectralOperator`
            Joined operator, generally larger in rank if
            `strength>0`, otherwise equal in rank to the
            parent eigenbasis.

        """

        if isinstance(operator,SpectralOperator) \
            and self.shares_basis(operator): # This may bypass the `augment` flag
                self.msg("Multiplying spectral operators in common basis...")
                eigvals = self.eigenvalues * operator.eigenvalues
                eigfuncs = self.eigenfunctions

                joined=SpectralOperator(zip(eigvals,eigfuncs),process=False)
                joined._inherit_basis(self) # Inherit hash label for basis

        else:
            if not augment:
                self.msg("Multiplying spectral operators in projected basis...")
                basis=self
                L1mat = self.as_matrix_in()
                if isinstance(operator,np.matrix): L2mat=operator
                else: L2mat = operator.as_matrix_in(self) #We have performed change of basis from 2 to 1

            else:
                assert isinstance(operator,SpectralOperator)
                self.msg("Multiplying spectral operators in augmented basis...")
                basis=self.augmented_basis(self,operator,smoothing=smoothing)
                L1mat=self.as_matrix_in(basis) #a matrix
                L2mat=operator.as_matrix_in(basis)

            if direction=='left': L3mat = L1mat @ L2mat
            else: L3mat = L2mat @ L1mat

            joined=basis.expand_matrix(L3mat)

        return joined

    def __getattribute__(self, name):

        if name=='eigenvalues': return np.array(list(self.keys()))
        elif name=='eigenfunctions': return list(self.values())
        elif name=='matrix': return self.as_matrix_in()

        return super().__getattribute__(name)

    def __add__(self,other):

        if isinstance(other, (SpectralOperator,np.matrix)):
            return self.add(other)

        #If number, just multiply eigenvalues
        elif isinstance(other, numbers.Number):

            if other==0: return self

            new_eigenvalues = self.eigenvalues + other

            #Wrap in new operator, but same basis
            new=SpectralOperator(zip(new_eigenvalues,\
                                     self.eigenfunctions),\
                                    process=False,\
                                    size=self.size)
            new._inherit_basis(self)

            return new

        else: return NotImplemented

    def __mul__(self,other):

        #Handle the non-commutativity
        if isinstance(other, (SpectralOperator,np.matrix)):
            return self.mul(other,direction='left') #left multiply by self

        #If number, just multiply eigenvalues
        elif isinstance(other, numbers.Number):

            if other==1: return self
            elif other==0: return 0

            new_eigenvalues = other * self.eigenvalues

            #Wrap in new operator, but same basis
            new=SpectralOperator(zip(new_eigenvalues,\
                                     self.eigenfunctions),\
                                    process=False,\
                                    size=self.size)
            new._inherit_basis(self)

            return new

        else: return NotImplemented

    #Addition should be commutative so this is ok
    def __radd__(self,other): return self.__add__(other)

    #We defer to the left-multiplication of
    def __rmul__(self,other):

        #Handle the non-commutativity
        if isinstance(other, (SpectralOperator,np.matrix)):
            return self.mul(other,direction='right') #left multiply by self

        #Otherwise, it's commutative and apply left multiplication
        else: return self.__mul__(other)

    def __sub__(self,other): return self.__add__(-1*other)

    def __rsub__(self,other): return (-1*self).__add__(other)

    #do we need `__rsub__?

    #Defer regular division to right division
    def __truediv__(self,other):

         #We know how to left-multiply by 1/other
         return self.__mul__(1/other) #If `other` is spectral operator, it will invoke `__rtruediv__`

    def __rtruediv__(self,other):

        #Handle the non-commutativity
        if isinstance(other, SpectralOperator):
            return NotImplemented('Ambiguous order of operations.')

        #If number, just multiply eigenvalues
        elif isinstance(other, numbers.Number):

            if other==0: return 0

            new_eigenvalues = other / self.eigenvalues

            #Wrap in new operator, but same basis
            new=SpectralOperator(zip(new_eigenvalues,\
                                     self.eigenfunctions),\
                                    process=False,\
                                    size=self.size)
            new._inherit_basis(self)

            return new
        
def load_spectral_operator(name='UnitSquareMesh_101x101x2000_Neumann_eigenbasis.pickle'):

    filepath=os.path.join(os.path.dirname(__file__),'..',\
                        'eigenbasis_library',name)
    print('Loading eigenpairs from "%s"...'%filepath)

    f=open(filepath,'rb')

    E=pickle.load(f); f.close()

    if isinstance(E,dict):
        eigpairs={}
        for eigval in E: eigpairs[eigval-1]=E[eigval]

    elif isinstance(E,AWA):
        eigpairs=[(E.axes[0][i],E[i]) for i in range(len(E))]

    else: raise NotImplementedError('Data type %s not understood.'%type(E))

    return SpectralOperator(eigpairs)

class SpectralLaplacian(SpectralOperator): pass #For inheritance only

class SpectralLaplacian_rect(SpectralLaplacian):

    def __init__(self,Lx=12,Rx=10,Ry=10,Nx=150,Ly=12,x0=0,y0=0,\
                 Nqmax=None,neumann=True,include_const=False):

        Rxmin,Rxmax=-Rx/2,+Rx/2
        Rymin,Rymax=-Ry/2,+Ry/2
    
        xv,yv=utils.get_XY((Lx,Ly),Nx,sparse=False)
        xs=xv[:,0]; ys=yv[0]
        Ny=len(ys)
    
        T=utils.Timer()
        print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))

        global eigpairs
        global eigmultiplicity

        eigpairs = {}
        eigmultiplicity = {}

        Nqsx=int(Nx*Rx/Lx/2) #No use in including eigenfunctions with even greater periodicity
        Nqsy=int(Ny*Ry/Ly/2) #No use in including eigenfunctions with even greater periodicity

        if neumann: phi0=np.pi/2
        else: phi0=0
        if Nqmax is None: Nqmax=Nqsx*Nqsy

        q0x=np.pi/Rx #This is for particle in box (allowed wavelength is n*2*L)
        q0y=np.pi/Ry #This is for particle in box (allowed wavelength is n*2*L)

        qxs=np.arange(Nqsx+1)*q0x
        qys=np.arange(Nqsy+1)*q0y
        qpairs=list(product(qxs,qys))
        eigvals=[qx**2+qy**2 for qx,qy in qpairs]
        eigvals,qpairs=zip(*sorted(zip(eigvals,qpairs)))
        eigvals=list(eigvals)[:Nqmax//2] #factor of 2 is because we entertain a 2-fold degeneracy
        qpairs=list(qpairs)[:Nqmax//2]

        for eigval,qpair in zip(eigvals,qpairs):

            #We cannot admit a constant potential, no charge neutrality
            if eigval==0: continue

            qx,qy=qpair
            if not neumann and (qx==0 or qy==0): continue #All must be nonzero for dirichlet

            pw1=utils.planewave(qx,0,xv,yv,\
                          x0=Rxmin+x0,y0=Rymin+y0,\
                          phi0=phi0)
            pw2=utils.planewave(0,qy,xv,yv,\
                          x0=Rxmin+x0,y0=Rymin+y0,\
                          phi0=phi0)
            pw = AWA(pw1*pw2, axes = [xs,ys], axis_names=['x','y'])

            pw[(xv<Rxmin+x0)]=0
            pw[(xv>Rxmax+x0)]=0
            pw[(yv<Rymin+y0)]=0
            pw[(yv>Rymax+y0)]=0
            pw-=np.mean(pw) #subtracting an accumulated offset is harmless, and good for bookkeeping

            while eigval in eigpairs: eigval+=1e-8
            eigpairs[eigval]=pw
                
        #Now include constant function
        if include_const:
            eigpairs[0]=utils.normalize(AWA(np.zeros(xv.shape)+1,\
                                      axes = [xs,ys], axis_names=['x','y']))
    
        print(T())

        return super().__init__(eigpairs)

class SpectralLaplacian_ribbon(SpectralLaplacian):

    def __init__(self,Lx=12,Rx=10,Nx=150,Ly=12,x0=0,\
                          Nqmax=None,qys=None,\
                          neumann=True,include_const=False): 

        Rxmin,Rxmax=-Rx/2,+Rx/2

        
        xv,yv=utils.get_XY((Lx,Ly),Nx,sparse=False)
        xs=xv[:,0]; ys=yv[0]
        Ny=len(ys)
        
        T=utils.Timer()
        print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))

        if neumann: phi0x=np.pi/2
        else: phi0x=0

        global eigpairs
        global eigmultiplicity

        eigpairs = {}
        eigmultiplicity = {}

        Nqsx=int(Nx*Rx/Lx/2) #No use in including eigenfunctions with even greater periodicity
        Nqsy=int(Ny/2) #No use in including eigenfunctions with even greater periodicity
        if Nqmax is None: Nqmax=Nqsx*Nqsy

        #make sure we have an even number of Nqsx and Nqsy
        if Nqsx%2: Nqsx+=1
        if Nqsy%2: Nqsy+=1

        q0x=np.pi/Rx #This is for particle in box (allowed wavelength is n*2*L)
        qxs=np.arange(0,Nqsx+1)*q0x

        if qys is None:
            q0y=2*np.pi/Ly #This is for periodic bcs (allowed wavelength is n*L)
            qys=np.arange(0,Nqsy+1)*q0y

        pairs=list(product(qxs,qys))
        eigvals=[qx**2+qy**2 for qx,qy in pairs]
        eigvals,qpairs=zip(*sorted(zip(eigvals,pairs)))
        eigvals=list(eigvals)[:Nqmax//2] #factor of 2 is because we entertain a 2-fold degeneracy
        qpairs=list(qpairs)[:Nqmax//2]

        for eigval,qpair in zip(eigvals,qpairs):

            #We cannot admit a constant potential, no charge neutrality
            if eigval==0: continue
            qx,qy=qpair

            #First the cos*sin wave
            pw1=utils.planewave(qx,0,xv,yv,\
                          x0=Rxmin+x0,y0=ys.min(),\
                          phi0=phi0x)
            pw2=utils.planewave(0,qy,xv,yv,\
                          x0=Rxmin+x0,y0=ys.min(),\
                          phi0=0)
            pw = AWA(pw1*pw2, axes = [xs,ys], axis_names=['x','y'])

            pw[(xv<Rxmin+x0)]=0
            pw[(xv>Rxmax+x0)]=0
            pw-=np.mean(pw) #This is to ensure charge neutrality

            if pw.any():
                while eigval in eigpairs: eigval+=1e-8
                eigpairs[eigval]=pw

            #Second the cos*cos wave
            pw2=utils.planewave(0,qy,xv,yv,\
                          x0=Rxmin+x0,y0=ys.min(),\
                          phi0=np.pi/2)
            pw = AWA(pw1*pw2, axes = [xs,ys], axis_names=['x','y'])

            pw[(xv<Rxmin+x0)]=0
            pw[(xv>Rxmax+x0)]=0
            pw-=np.mean(pw) #subtracting an accumulated offset is harmless, and good for bookkeeping

            if pw.any():
                while eigval in eigpairs: eigval+=1e-8
                eigpairs[eigval]=pw
                
        #Now include constant function
        if include_const:
            eigpairs[0]=utils.normalize(AWA(np.zeros(xv.shape)+1,\
                                      axes = [xs,ys], axis_names=['x','y']))
    
        print(T())

        return super().__init__(eigpairs)

class SpectralLaplacian_uniform(SpectralLaplacian):

    def __init__(self,Lx=10,Nx=100,Ly=10,Nqmax=None,qys=None,include_const=False):

        dx=Lx/Nx
        xs=np.arange(Nx)*dx; xs-=np.mean(xs)
        Ny=int(Ly/Lx*Nx)
        ys=np.arange(Ny)*dx; ys-=np.mean(ys)

        T=utils.Timer()
        print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))

        global eigpairs
        global eigmultiplicity

        yv,xv = np.meshgrid(ys,xs)
        eigpairs = {}
        eigmultiplicity = {}

        Nqsx=int(Nx/4) #No use in including eigenfunctions with even greater periodicity
        Nqsy=int(Ny/4) #No use in including eigenfunctions with even greater periodicity

        if Nqmax is None: Nqmax=Nqsx*Nqsy


        q0x=2*np.pi/Lx #This is for periodic bcs (allowed wavelength is n*L)
        qxs=np.arange(Nqsx+1)*q0x

        if qys is None:
            q0y=2*np.pi/Ly #This is for periodic bcs (allowed wavelength is n*L)
            qys=np.arange(Nqsy+1)*q0y

        qpairs=list(product(qxs,qys))
        eigvals=[qx**2+qy**2 for qx,qy in qpairs]
        eigvals,qpairs=zip(*sorted(zip(eigvals,qpairs)))
        eigvals=list(eigvals)[:Nqmax//4] #factor of 4 is because we entertain a 4-fold degeneracy
        qpairs=list(qpairs)[:Nqmax//4]

        for eigval,qpair in zip(eigvals,qpairs):

            #We cannot admit a constant potential, no charge neutrality
            if eigval==0: continue
            qx,qy=qpair

            #First the cos*sin wave
            pwxs=utils.planewave(qx,0,xv,yv,\
                          x0=np.min(xs),y0=np.min(ys),\
                          phi0=0)
            pwxc=utils.planewave(qx,0,xv,yv,\
                          x0=np.min(xs),y0=np.min(ys),\
                          phi0=np.pi/2)
            pwys=utils.planewave(0,qy,xv,yv,\
                          x0=np.min(xs),y0=np.min(ys),\
                          phi0=0)
            pwyc=utils.planewave(0,qy,xv,yv,\
                          x0=np.min(xs),y0=np.min(ys),\
                          phi0=np.pi/2)

            for pw in [pwxc*pwyc,pwxs*pwyc,pwxc*pwys,pwxs*pwys]:

                if not pw.any(): continue

                pw = AWA(pw, axes = [xs,ys], axis_names=['x','y'])
                pw-=np.mean(pw) #subtracting an accumulated offset is harmless, and good for bookkeeping

                while eigval in eigpairs: eigval+=1e-8
                eigpairs[eigval]=utils.normalize(pw)

        #Now include constant function
        if include_const:
            eigpairs[0]=utils.normalize(AWA(np.zeros(xv.shape)+1,\
                                      axes = [xs,ys], axis_names=['x','y']))

        print(T())

        return super().__init__(eigpairs)

class SpectralLaplacian_disk(SpectralLaplacian):

    def __init__(self,Lx=10,Nx=100,Ly=10,Nqmax=2000,lmax=10,include_const=False,\
                 r0=(0,0),sort=True):

        from scipy.special import jv,jn_zeros

        dx=Lx/Nx
        xs=np.arange(Nx)*dx; xs-=np.mean(xs)
        Ny=int(Ly/Lx*Nx)
        ys=np.arange(Ny)*dx; ys-=np.mean(ys)
        yv,xv = np.meshgrid(ys,xs,sparse=True)
        x0,y0=r0
        rv=np.sqrt((xv-x0)**2+(yv-y0)**2)
        #q0=2*np.pi/(rv.max()) # max radius will be our longest wavelength
        a=rv.max()
        thetav=np.arctan2(yv-y0,xv-x0)

        T=utils.Timer()
        print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))

        if lmax is None:
            nmax=int(Nx/4) #No use in including eigenfunctions with even greater periodicity
            lmax=int(round(Nqmax/nmax/2))
        else:
            nmax=int(round(Nqmax/lmax/2))
            
        print('nmax=',nmax,'lmax=',lmax)
        
        global eigpairs
        eigpairs={};
        for l in range(0,lmax):

            lambdas=jn_zeros(l, nmax)

            for n in range(0,nmax):
                #qn=n*q0; eigval=qn**2 #This is an approximation, we should actually be selecting n'th root of jn(l,...)

                qn=lambdas[n]/a
                eigval=qn**2
                jradial=jv(l,qn*rv)*(-1)**n

                wave1=jradial*np.cos(l*thetav)
                wave1-=np.mean(wave1)

                while eigval in eigpairs: eigval+=1e-8

                wave1 = AWA(wave1, axes = [xs,ys], axis_names=['x','y'])
                eigpairs[eigval]=utils.normalize(wave1)

                if l>0:
                    wave2=jradial*np.sin(l*thetav)
                    wave2-=np.mean(wave2)

                    while eigval in eigpairs: eigval+=1e-8
                    wave2 = AWA(wave2, axes = [xs,ys], axis_names=['x','y'])
                    eigpairs[eigval]=utils.normalize(wave2)

        #Now include constant function
        if include_const:
            eigpairs[0]=utils.normalize(AWA(np.zeros(xv.shape)+1,\
                                      axes = [xs,ys], axis_names=['x','y']))

        print(T())

        #Sort the eigenpairs by eigenvalue
        if sort:
            Es,fs=list(zip(*eigpairs.items()))
            Es=np.array(Es)

            idx=np.argsort(Es)
            Es=Es[idx]
            fs=[fs[i] for i in idx]
            eigpairs=zip(Es,fs)

        return super().__init__(eigpairs,process=False)

#--- `GridOperator` and subclasses

class GridOperator(object):
    """Totally inert until subclassed."""
    
    debug=True

    @classmethod
    def msg(cls,msg):
        if cls.debug: print(msg)

    def __init__(self,grid_operation,dx=1,order=2):

        self._grid_operation=grid_operation
        self.dx=dx
        self._order=order

    def __call__(self,*args,**kwargs): raise NotImplementedError

    def as_matrix_in(self,basis):

        if not isinstance(basis,Basis): basis=Basis(basis)

        #Technically this result is like `<basis| self*|basis>`
        # which we calculate using ` conj( <self*basis | basis> ).T`
        M_hc = basis.project_into(self(basis.functions)) #This is hermitian transpose of what we want
        
        return M_hc.H #This applies Hermitian transpose

    def as_SpectralOperator(self,basis,**kwargs):
        """
        Builds a spectral representation for `grid_operator` by
        expanding first in `grid_basis`, then diagonalizing to
        find an appropriate eigenbasis.
        """

        #Make a projector operator that can expand in `basis`
        if isinstance(basis,SpectralOperator): projector=basis
        else:
            if isinstance(basis,Basis): basis=basis.functions #We want a list of functions
            projector=SpectralOperator([(1,eigfunc) for eigfunc in basis],\
                                       **kwargs)

        M=self.as_matrix_in(basis)

        return projector.expand_matrix(M)

    def __add__(self,other):

        #To use any `SpectralOperator` we have to assume we're Hermitian
        if isinstance(other, SpectralOperator):

            #seamlessly add with `__add__` of `SpectralOperator`
            return other.__add__(self.as_matrix_in(other.eigenfunctions))

        #If number, just modify `grid_operation`
        elif isinstance(other, numbers.Number):
            new=copy.copy(self)
            def new_grid_operation(*args,**kwargs):
                return other+self._grid_operation(*args,**kwargs)

            new._grid_operation=new_grid_operation
            return new

    #Without an explicit basis, addition is always commutative
    def __radd__(self,other): return self.__add__(other)

    def __mul__(self,other):

        #To use any `SpectralOperator` we have to assume we're Hermitian
        if isinstance(other, SpectralOperator):

            #hope parent implements `__rmul__`
            return self.as_matrix_in(other.eigenfunctions) * other

        #If number, just modify `grid_operation`
        elif isinstance(other, numbers.Number):
            new=copy.deepcopy(self)
            def new_grid_operation(*args,**kwargs):
                result=self._grid_operation(*args,**kwargs)
                return [other*f for f in result]

            new._grid_operation=new_grid_operation
            return new

        else: return NotImplemented('Multiplication with type %s not understood.'%type(other))

    def __rmul__(self,other):

        #To use any `SpectralOperator` we have to assume we're Hermitian
        if isinstance(other, SpectralOperator):

            #hope parent implements `__mul__`
            return other * self.as_matrix_in(other.eigenfunctions)

        #If number, multiplication is commutative
        elif isinstance(other, numbers.Number): return self.__mul__(other)

        else: return NotImplemented('Multiplication with type %s not understood.'%type(other))

    def __truediv__(self,other): return self.__mul__(1/other) #leave it to `other` to refine `__rtruediv__`

    def __sub__(self,other): return self.__add__(-1*other)

    def __rsub__(self,other): return (-1*self).__add__(other)

class GridDifferentialOperator(GridOperator):

    def __call__(self,functions):
        """Assume the grid operation can handle a list of functions,
        so pass them right through the operation while dressing with dx.
        This means that input `AWA`s will generate output `AWA`s."""

        #Make sure we have a list of functions
        functions=np.asarray(functions)
        if functions.ndim==2:
            functions=functions.reshape((1,)+functions.shape)

        dfs=self._grid_operation(functions)
        dfsdx=[np.asanyarray(df)/self.dx**self._order for df in dfs]

        if len(dfsdx)==1: return dfsdx[0]
        else: return dfsdx

class GridIntegralOperator(GridOperator):

    def __call__(self,functions):
        """Assume the grid operation can handle a list of functions,
        so pass them right through the operation while dressing with dx.
        This means that input `AWA`s will generate output `AWA`s."""

        #Make sure we have a list of functions
        functions=np.asanyarray(functions)
        if functions.ndim==2:
            functions=functions.reshape((1,)+functions.shape)

        fs=self._grid_operation(functions)
        Fs=[np.asanyarray(f)*self.dx**self._order for f in fs]

        if len(Fs)==1: return Fs[0]
        else: return Fs

class GridLaplacian_aperiodic(GridDifferentialOperator):

    def __init__(self,dx,sigma_xx=1,sigma_yy=None,sigma=None):

        self.set_sigma(sigma_xx,sigma_yy,sigma)
        self.dx=dx
        self._order=2

    def set_sigma(self,sigma_xx=1,sigma_yy=None,sigma=None):

        #If `sigma` is actually provided, use it to override the others
        if sigma is not None: sigma_xx=sigma_yy=sigma
        if sigma_yy is None: sigma_yy=sigma_xx

        sigma_xx=np.asarray(sigma_xx).copy() #Copy because we will change shape
        sigma_xx.resize((1,)+sigma_xx.shape)
        while sigma_xx.ndim<3: sigma_xx.resize(sigma_xx.shape+(1,))

        sigma_yy=np.asarray(sigma_yy).copy() #Copy because we will change shape
        sigma_yy.resize((1,)+sigma_yy.shape)
        while sigma_yy.ndim<3: sigma_yy.resize(sigma_yy.shape+(1,))

        self.sigma_xx=sigma_xx
        self.sigma_yy=sigma_yy

    def _grid_operation(self,functions):
        "Wraps `cls.gridlaplacian_aperiodic` with user-defined `sigma`"

        return self.gridlaplacian_aperiodic(functions,self.sigma_xx,self.sigma_yy)

    @classmethod
    def gridlaplacian_aperiodic(cls,functions,sigma_xx,sigma_yy):
        """This guy is pretty inaccurate I think, it definitely gives
        inferior results to `Laplacian_periodic`."""
        
        eo=2

        #Make assume we have a list of functions
        functions=np.asarray(functions)

        cls.msg('Computing aperiodic Laplacian...')
        T=utils.Timer()

        dux=np.gradient(functions,axis=1,edge_order=eo)
        dux=np.gradient(sigma_xx*dux,axis=1,edge_order=eo)
        
        if sigma_yy.any():
            duy=np.gradient(functions,axis=2,edge_order=eo)
            duy=np.gradient(sigma_yy*duy,axis=2,edge_order=eo)
        else: duy=0

        Lu=-(dux+duy) #This gives us positive eigenvalues
        cls.msg(T())

        return Lu

class GridLaplacian_periodic(GridLaplacian_aperiodic):

    def _grid_operation(self,functions):
        "Wraps `cls.gridlaplacian_aperiodic` with user-defined `sigma`"

        return self.gridlaplacian_periodic(functions,self.sigma_xx,self.sigma_yy)

    @classmethod
    def gridlaplacian_periodic(cls,functions,sigma_xx,sigma_yy):
        """
        Apply the position-dependent laplace operator to a list of
        (assumed) 2D periodic `functions` on a mesh of size `Lx,Ly`.
        Position-dependent conductivity `sigma` should be specified
        as an array of shape matching each function in `functions`.

        For some extensions to a spatially varying derivative:
            "Algorithm 3" of the great Dr. Steven G. Johnson
            https://math.mit.edu/~stevenj/fft-deriv.pdf
        """

        #Make sure we have a list of functions
        functions=np.asarray(functions)

        Nx,Ny=functions[0].shape
        fx=np.fft.fftfreq(Nx,d=1).reshape((1,Nx,1)) #any units of will be handled later
        fy=np.fft.fftfreq(Ny,d=1).reshape((1,1,Ny))
        indxmin=Nx//2; indymin=Ny//2

        Savx=np.mean(sigma_xx,axis=1)
        Savy=np.mean(sigma_yy,axis=2)

        cls.msg('Computing periodic Laplacian...')
        T=utils.Timer()

        u=np.asarray(functions)
        Uxs=np.fft.fft(u,axis=1); Uxs_fmin=Uxs[:,indxmin,:]
        Uys=np.fft.fft(u,axis=2); Uys_fmin=Uys[:,:,indymin]

        Vx = np.fft.ifft(2*np.pi*1j*fx*Uxs,axis=1)
        Vy = np.fft.ifft(2*np.pi*1j*fy*Uys,axis=2)
        del Uxs,Uys
        if not np.iscomplexobj(Savx) or np.iscomplexobj(Savy):
            Vx=np.real(Vx)
            Vy=np.real(Vy)

        Vxs=np.fft.fft(sigma_xx*Vx,axis=-2)
        Vys=np.fft.fft(sigma_yy*Vy,axis=-1)
        del Vx,Vy

        # if N even
        # This step preserves self-adjointness of L operator, according to Dr. Johnson
        # The reasoning is above my paygrade
        if not Nx%2: Vxs[:,indxmin,:]= -Savx*(np.pi)**2*Uxs_fmin #watch out because we removed a 1/dx
        if not Ny%2: Vys[:,:,indymin]= -Savy*(np.pi)**2*Uys_fmin

        d2udx2 = np.fft.ifft(2*np.pi*1j*fx*Vxs,axis=1)
        d2udy2 = np.fft.ifft(2*np.pi*1j*fy*Vys,axis=2)
        del Vxs,Vys
        if not np.iscomplexobj(Savx) or np.iscomplexobj(Savy):
            d2udx2=np.real(d2udx2)
            d2udy2=np.real(d2udy2)

        Lu=-(d2udx2+d2udy2) #This gives us positive eigenvalues
        cls.msg(T())

        return Lu

#--- `GeneralOperator` and subclasses

class GeneralOperator(Basis):

    def __init__(self,operators,basis,\
                 operation=lambda *operators: builtins.sum(operators),\
                  **kwargs):

        #Set our basis using inherited basis constructor
        super(GeneralOperator,self).__init__(basis,**kwargs)

        self._operator_matrices=[]

        for operator in operators:
            self.add_operator(operator)

        self.operation=operation

    def add_operator(self,operator):

        n=len(self._operator_matrices)+1
        if hasattr(operator,'as_matrix_in'):
            self.msg('Projecting operator %i onto basis...'%n)
            m=utils.NormalMatrix(operator.as_matrix_in(self))

        #wrap any non-Matrix matrices to play nicely with Matrices
        elif isinstance(operator,np.matrix): m=utils.NormalMatrix(operator)

        #wrap any non-Constant numbers to play nicely with Matrices
        elif isinstance(operator,numbers.Number):
            if isinstance(operator,utils.Constant): m=operator
            else: m=utils.Constant(operator)

        else: raise ValueError('Operator type %s not understood!'%type(operator))

        self._operator_matrices.append(m)

    def get_matrix(self):
        """Form a matrix for the provided operation using the constructed matrices."""

        return self.operation(*self._operator_matrices)

    def get_operator_matrices(self):

        return copy.copy(self._operator_matrices)

    def __call__(self,functions):

        Vin=self.vectors_from(functions)

        Vout=self.matrix * Vin

        return self.expand_vectors(Vout)

    #--- Manipulations between different bases

    def as_matrix_in(self,basis=None):
        """
        Build a matrix representation for `self` projected into `basis`,
        specified by 1) a list of 2D array functions, 2) column vectors,
        or 3) another `SpectralOperator`.

        Meaningful if and only if `basis` is orthogonal.
        """

        if basis is None \
            or (isinstance(basis,Basis) and self.shares_basis(basis)):
                return np.matrix(np.diag(self.eigenvalues))

        #This corresponds to `<self|basis>`
        # which we obtain as `<basis|self>.T`
        P=self.project_into(basis)

        return P * self.matrix * np.conj(P).T #projects from `basis` into ourselves and back out

    #An alias
    M=as_matrix_in

    def get_eigpairs(self,as_dict=True):
        """
        Produce an eigenfunction description of this operator.

        Returns:
            With `as_dict=True`, dictionary of eigenvalue / right eigenfunction pairs.
            
            Otherwise, tuple of `eigvals, right eigenfunctions, left eigenfunctions`.

        @ASM 2020.10.05: Improve the abstraction of returned results, for left/right eigenmodes
        """

        M=self.matrix

        if np.iscomplexobj(M): eig=np.linalg.eig; Mtype='non-Hermitian'
        else: eig=np.linalg.eigh; Mtype='Hermitian'

        self.msg("Diagonalizing new %s operator of size %s..."%(Mtype,str(M.shape)))
        T=utils.Timer()
        eigvals,eigvecsR=eig(M)
        #If right eigenvectors are orthonormal, then `.I` and `.H` are the same, and left/right eigenvectors are the same
        eigvecsL=eigvecsR.H.I

        #expand the linear combinations of columns represented by eigenvectors
        self.msg("Expanding right and left eigenfunctions...")
        eigfuncsR=self.expand_vectors(eigvecsR)
        eigfuncsL=self.expand_vectors(eigvecsL)
        self.msg(T())

        if as_dict: return dict(zip(eigvals,eigfuncsR))
        else:
            eigvals,eigfuncsR,eigfuncsL=zip(*sorted(zip(eigvals,eigfuncsR,eigfuncsL),\
                                         key=lambda trio: np.real(trio[0])))
            return eigvals,eigfuncsR,eigfuncsL

    def __getattribute__(self, name):

        if name=='matrix': return self.get_matrix()
        elif name=='eigpairs': return self.get_eigpairs()

        return super(GeneralOperator,self).__getattribute__(name)

# -*- coding: utf-8 -*-
