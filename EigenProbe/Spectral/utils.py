import os,h5py
import time
import pickle
import numpy as np
import numbers
from scipy import special as sp
from functools import wraps
import warnings
from common.baseclasses import AWA

def get_argnames(cls):
    cls_co = cls.__init__.__code__
    nargs = cls_co.co_argcount
    return cls_co.co_varnames[:nargs]

def align_to_reals(psi0):
    """This algorithm applies an overall phase to align
    a complex vector with the real axis, in an average sense."""

    psi0=np.asarray(psi0)
    R2=np.sum(psi0**2)/np.sum(np.conj(psi0)**2)

    # Phase is defined only up to pi/2
    p=1/4*np.angle(R2); psi=psi0*np.exp(-1j*p)

    #Check if we chose correct phase
    Nr=np.real(np.sum(np.abs(psi+np.conj(psi))**2)/4)
    N=np.real(inner_prod(psi,psi))
    Ni=np.real(np.sqrt(N**2-Nr**2))
    realness=Nr/Ni

    if Ni and realness<1: psi*=np.exp(-1j*np.pi/2)

    return psi

def build_matrix(functions1,functions2):
    """
    Builds matrix of inner products between two lists of functions.
    TODO: this function is where all the meat is, so probably need to optimize.

    Parameters
    ----------
    functions1 : list of 2D `np.ndarray`
        Represents a basis of functions.
    functions2 : list of 2D `np.ndarray`
        Represents a basis of functions.

    Returns
    -------
    M_mn : `np.matrix`
        Element m,n corresponds with inner product
        < `functions1[m]` | `functions2[n]` >.

    """

    Nfunctions1=len(functions1)
    Nfunctions2=len(functions2)
    #assert len(functions1)==len(functions2)

    print('Building %ix%i matrix...'%(Nfunctions1,Nfunctions2))

    U1=np.array([func.ravel() for func in functions1]).T #column vectors
    U2=np.array([func.ravel() for func in functions2]).T

    M_mn = np.matrix(np.conj(U1).T @ U2)

    return M_mn

def dipole_field(x,y,z,direction=[0,1]):
    "`direction` is a vector with `[\rho,z]` components"

    r=np.sqrt(x**2+y**2+z**2)
    rho=np.sqrt(x**2+y**2)
    rhat_rho=rho/r
    rhat_z=z/r

    return (direction[0]*rhat_rho+direction[1]*rhat_z)/r**2

def unscreened_coulomb_kernel_fourier(kx,ky):
    q=np.sqrt(kx**2+ky**2)
    return np.where(q==0,0,2*np.pi/q)

def bessel(A,v,Q,x,y):
    r = np.sqrt(x**2+y**2)
    return A*sp.jv(v,Q*r)

def planewave(qx,qy,x,y,x0=0,y0=0,phi0=0):
    return np.sin(qx*(x-x0)+qy*(y-y0)+phi0)

#--- Utilities for `Basis`

class Timer(object):

    def __init__(self):

        self.t0=time.time()

    def __call__(self,reset=True):

        t=time.time()
        msg='\tTime elapsed: %s'%(t-self.t0)

        if reset: self.t0=t

        return msg

def inner_prod(psi1,psi2,metric=1):

    psi1=np.asarray(psi1)
    psi2=np.asarray(psi2)

    return np.sum(np.conj(psi1)*metric*psi2)

def norm(psi,metric=1):
    
    return np.sqrt(inner_prod(psi,psi,metric))

def normalize(psi,metric=1):
    """
    Normalize an `numpy.ndarray` function or
    (matrix of) vector(s) according to its norm.
    
    @TODO: generalize normalization with a metric"""
    
    #Assume we have one or more vectors to normalize
    if isinstance(psi,np.matrix):
        for v in psi.T: v[:]=v/norm(v,metric)
        return psi
    
    #In case we receive a sequence of functions
    psitest=np.asarray(psi)
    if psitest.ndim==3: return [normalize(p) for p in psi]

    return psi/norm(psi,metric)

#--- Utilities for `SpectralOperator`

def is_square(m):
    return m.shape[0] == m.shape[1]

def is_normal(m):
    m=np.matrix(m,copy=False)
    return np.allclose(m * m.H, m.H * m)

def get_XY(size,Nx,center=(0,0),sparse=True):
    """
    Return a meshgrid (`sparse` or otherwise) of x,y coordinates
    pursuant to a rectangular simulation of size `size` with `Nx`
    pixels along the x-coordinate.  Y-pixels are produced on the
    basis of square pixels.  `center` allows to displace the center
    coordinate from `(0,0)` (default).
    
    @TODO 2020.06.07: merge into a common `utils.py` for `Spectral`"""
    
    Lx,Ly=size
    x0,y0=center
    
    dx=Lx/Nx
    xrange=dx*(Nx-1)
    x=np.linspace(-xrange/2+x0,xrange/2+x0,Nx)
    
    Ny=int(Ly/Lx*Nx)
    dy=Ly/Ny
    yrange=dy*(Ny-1)
    y=np.linspace(-yrange/2+y0,yrange/2+y0,Ny)
    
    Y,X=np.meshgrid(y,x,sparse=sparse)
    
    return X,Y

#--- `Utilities for `GeneralOperator`

class Constant(np.complex):
    """
    This is a duck-type for a complex constant, except we can
    update the value remotely.  References will reflect that value
    at runtime.

    This makes the object behave like a mutable operator.

    It also behaves nicely with matrices (but this is probably redundant).
    """

    def __new__(cls, value, value_im=0):
        """This constructor should be compatible with the call signature for 
        the constructor for `np.complex`, with the added feature of "passing through"
        any existing objects of type `Constant`."""
        
        if value_im: value=value+1j*value_im

        if not isinstance(value,cls):
            obj = np.complex.__new__(cls, value)
            obj.set_value(value)
        else: obj=value

        return obj
    
    # def __reduce__(self):
        
    #     tup=list(super().__reduce__())
    #     tup[1]=(self.get_value(),)
        
    #     return tuple(tup)

    def set_value(self,value):

        assert isinstance(value,numbers.Number)
        self._value=np.complex(value) #Must be complex type, because this is our inheritence

    def get_value(self): return self._value

    #--- Decorate some operations to respect Constant value
    def _delegated_to_value_(method_name):
        """This will take an operator name and produce
        a function that looks like the operator of that
        name attached to an `np.complex`."""

        method=getattr(np.complex,method_name)

        @wraps(method,assigned=('__name__','__doc__'))
        def inherited_operator(self): return method(self.get_value())

        return inherited_operator

    _repr_ops_=['__repr__',\
                  '__str__']
    for op in _repr_ops_: locals()[op]=_delegated_to_value_(op)

    def _respect_Constant_and_ndarray_(method_name):
        """Ensure that `Constant` values are respected, and
        poperly delegates operations to `numpy.ndarray` types.
        
        Do not implement operation on `NormalMatrix`, that is
        left to that class definition."""

        method=getattr(np.complex,method_name)

        @wraps(method,assigned=('__name__','__doc__'))
        def respectful_method(self,other):
            
            #To delegate to `ndarray` types is to assert commutativity, the definition of a constant
            if isinstance(other,np.ndarray):
                 other_method=getattr(other,method_name)
                 return other_method(self.get_value())

            if isinstance(other,Constant): other=other.get_value()

            result=method(self.get_value(),other)

            ##If not a Number subclass, we're done##
            if not isinstance(result,numbers.Number): return result

            ##Otherwise, cast as self type##
            else: return type(self)(result)

        return respectful_method

    _binary_ops_=['__add__',\
                    '__sub__',\
                    '__mul__',\
                    '__truediv__',
                    '__pow__',\
                    '__mod__',\
                    '__divmod__',\
                    '__floordiv__']
    _binary_rops_=['__radd__',\
                    '__rsub__',\
                    '__rmul__',\
                    '__rtruediv__',\
                    '__rpow__',\
                    '__rmod__',\
                    '__rdivmod__',\
                    '__rfloordiv__']
    _delegate_ops_ = _binary_ops_ + _binary_rops_
    for op in _delegate_ops_: locals()[op]=_respect_Constant_and_ndarray_(op)
    del op

class NormalMatrix(np.matrix):

    def __new__(cls, value):

        if not isinstance(value,cls):
            obj=np.asmatrix(value).view(cls)
        else: obj=value

        assert is_square(obj),"Shape must be square!"

        return obj

    def __truediv__(self,other):
        """Override to treat matrices in the sense of inverse.
        This is the whole point of this class.
        
        Any further operations get kicked to `__mul__`.

        Tested and works."""

        if isinstance(other,np.matrix): return self * other.I

        else: return self * (1/other) #If `other` doesn't define right division, not our fault.


    def __rtruediv__(self,other):
        """Override to treat matrices in the sense of inverse.
        This is the whole point of this class.
        
        Any further operations get kicked to `__mul__`.

        Tested and works."""

        warnings.warn('Ambiguous order of operations in matrix division!')

        return self.I.__mul__(other)

    #--- Decorate the behavior of certain operations to respect Constant value
    def _respects_Constants_(method_name):
        """Ensure that `Constant` values are respected, and that
        numeric types are treating like multiples of identity."""
        
        method=getattr(np.matrix,method_name)
        
        #This flag activates an additional clause in the wrapped method
        expand_numeric=(method_name in ['__add__','__radd__','__sub__','__rsub__'])

        @wraps(method,assigned=('__name__','__doc__'))
        def respectful_method(self,other):

            if isinstance(other,Constant): other=other.get_value()
            
            #For certain operations (`__add__`, etc.), wrapper gets instruction
            # to expand numeric types to multiple of identity matrix
            if isinstance(other,numbers.Number) and expand_numeric:
                other=other*np.matrix(np.eye(len(self)))

            result=method(self,other)

            ##If not a matrix subclass, we're done##
            if not isinstance(result,np.matrix): return result

            ##If a square matrix, cast as self type##
            elif is_square(result): return type(self)(result)
            
            return result

        return respectful_method

    #--- Decorate all these binary operations
    _binary_ops_=['__add__',\
                    '__divmod__',\
                    '__floordiv__',\
                    '__mod__',\
                    '__mul__',\
                    '__pow__',\
                    '__sub__']
    _binary_rops_=['__radd__',\
                    '__rdivmod__',\
                    '__rfloordiv__',\
                    '__rmod__',\
                    '__rmul__',\
                    '__rpow__',\
                    '__rsub__']
    for op in _binary_ops_ + _binary_rops_:
        locals()[op]=_respects_Constants_(op)
    del op
