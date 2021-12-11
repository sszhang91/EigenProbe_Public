import time
import numpy as np

def progress(i,L,last):
    next = last+10
    percent = 100*i/L
    if percent >= next:
        print('\t{}% complete...'.format(next))
        return next
    else:
        return last

class Timer(object):

    def __init__(self):

        self.t0=time.time()

    def __call__(self,reset=True):

        t=time.time()
        msg='\tTime elapsed: %s'%(t-self.t0)

        if reset: self.t0=t

        return msg
    
#--- Linear algebra tools, largely deprecated in favor of those in `Spectral`

def inner_prod(psi1,psi2):

    psi1=np.asarray(psi1)
    psi2=np.asarray(psi2)

    return np.sum(np.conj(psi1)*psi2)

def normalize(psi):

    return psi/np.sqrt(inner_prod(psi,psi))

# This kind of makes `inner_prod` redundant
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

    T=Timer()
    U1=np.array([func.ravel() for func in functions1]).T #column vectors
    U2=np.array([func.ravel() for func in functions2]).T

    M_mn = np.matrix(np.conj(U1).T @ U2)
    T()

    return M_mn

#--- Book-keeping tools

def align_to_reals(psi0):
    """This algorithm applies an overall phase to align
    a complex vector with the real axis, in an average sense."""

    psi0=np.asanyarray(psi0)
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

def ensure_unique(seq,eps=1e-8):
    """Ensure all elements in `seq` are unique, by adding sufficient multiples of `eps` to repeated items.
    
    This gets utilized in `SubstrateFromLayers` to appropriate unique eigenvalues from `rp`."""
    
    seq=list(seq)
    for i in range(len(seq)):
        while seq[i] in set(seq[:i]+seq[i+1:]): seq[i]+=eps
        
    return seq

#--- Fields and eigenoscillations

def get_size(awa):
    
    ax_lims=awa.axis_limits
    size=[np.abs(np.diff(ax_lims[i])) \
          for i in range(2)]
        
    return size

def invert_axes(awa):
    """Invert the axes of an `AWA` across the origin `(x,y,...)=0.
    
    This is used to transform a probe field into a convolution kernel
    that's suitable for describing a raster-scan."""
    
    flipped_awa=awa.copy()
    flipped_awa.set_axes([-ax for ax in flipped_awa.axes])
    
    return flipped_awa.sort_by_axes()

def dipole_field(x,y,z,direction=[0,1]):
    "`direction` is a vector with `[\rho,z]` components"

    r=np.sqrt(x**2+y**2+z**2)
    rho=np.sqrt(x**2+y**2)
    rhat_rho=rho/r
    rhat_z=z/r

    return (direction[0]*rhat_rho+direction[1]*rhat_z)/r**2

def planewave(qx,qy,x,y,x0=0,y0=0,phi0=0):
    return np.sin(qx*(x-x0)+qy*(y-y0)+phi0)

def damped_bessel_eigenoscillation(n,xs,ys,Q=2*np.pi):
    """Characteristic "wavelength" of 1."""

    from scipy.special import j0
    q = 2*np.pi*n #this along with x,y are in units of tip size; also assume n>=1

    #The larger Q goes, the shorter range the tip excitation
    
    rs=np.sqrt(xs**2+ys**2)
    
    return np.exp(-q/Q*rs**2)\
                    *j0(q*rs)
                
#--- Coulomb kernels

def unscreened_coulomb_kernel_fourier(kx,ky):
    q=np.sqrt(kx**2+ky**2)
    return np.where(q==0,0,2*np.pi/q)

def unscreened_coulomb_kernel_retarded_fourier(kx,ky,q0):
    
    q=np.sqrt(kx**2+ky**2)
    kz=np.sqrt(q0**2-q**2+0j) #0j ensures complex output
    
    return np.where(q==0,0,-2*np.pi*1j*kz/q**2)

def unscreened_coulomb_kernel_retarded_fourier2(kx,ky,q0):
    
    q=np.sqrt(kx**2+ky**2)
    kz=np.sqrt(q0**2-q**2+0j) #0j ensures complex output
    
    return np.where(q==0,0,-2*np.pi/(1j*kz))


