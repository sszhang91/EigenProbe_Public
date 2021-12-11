import copy
import numpy as np
import Spectral as S
import EigenProbe.tip_modeling as TM

from NearFieldOptics.Materials import *
from NearFieldOptics.Materials.material_types import *
from NearFieldOptics.Materials.TransferMatrixMedia import MatrixBuilder,Calculator
from matplotlib import pyplot as plt
from common import numerical_recipes as numrec
from common.baseclasses import AWA
from common.numerical_recipes import QuickConvolver,smooth

import warnings
warnings.filterwarnings('ignore')

def SDai_hBN_results():
    #--- Define mutable constant for substrate reflectivity
    beta=S.Constant(.5)

    #--- Define wavelength and Q-factor (will encodes baseline conductivity for our uniform laplacian)
    WL=S.Constant(3)           #plasmon wavelength
    sigma2D=S.Constant(1-.01j)
    sigmaPML = S.Constant(-30j)

    #--- Build rectangular laplacian with edge for graphene (particle-in-box planewaves)
    N=300; L=200; Nq=100; Rx=0.8*L; Dx=(L-Rx)/2

    Graphene=S.SpectralLaplacian_ribbon(Lx=L,Nx=N,Ly=L,Nqmax=Nq,Rx=Rx,x0=Dx)
    X,Y=Graphene.XY
    x,y=Graphene.xy

    #--- Build homogeneous laplacian operator just to get its basis for the substrate (periodic plane waves)
    Substrate=S.SpectralLaplacian_uniform(Lx=L,Nx=N,Ly=L,Nqmax=Nq)
    Sample=Substrate.add(Graphene,augment=True)
    dx=np.diff(Sample.xy[0])[0]

    #--- Reduce the size of the total basis
    qmin=.1*(2*np.pi/L)
    Sample=Sample.filtered_by_eigenvalues(lambda E: (E>=qmin**2))
    qmax=np.sqrt(Sample.eigenvalues[Nq])
    Sample=Sample.filtered_by_eigenvalues(lambda E: (E<=qmax**2))
    Sample[0]=np.zeros(Sample.shape)+1 #Add in a constant function
    print('Size of combined basis:',len(Sample))

    top_hBN = (BN_Caldwell,5e-7)
    bottom_hBN = (BN_Caldwell,25e-7)
    layers = LayeredMediaTM(
                    top_hBN,
                    SingleLayerGraphene(),
                    bottom_hBN,
                    exit=SiO2_300nm);
    V = TM.CoulombOperatorFromLayers(layers,shape=Graphene.shape, size=Graphene.size);

    #--- Build Perfectly matched layer (PML)
    # Actually, it's not perfectly matched at all, just a border of absorbing conductivity
    PML=build_PML(dx,X,Y,L,sigmaPML,Sample)

    G2D=S.GeneralOperator((beta,V,sigma2D,Graphene,sigmaPML,PML,WL),basis=Sample,\
                         operation=Goperation2D) #Our Green's function

    G2D_0=S.GeneralOperator((beta,V,sigma2D,Graphene,sigmaPML,PML,WL),basis=Graphene,\
                             operation=Goperation2D_0) #Our Green's function
    GSubs=copy.copy(G2D) #Same operators, same basis, so just copy
    GSubs.operation=GoperationSubs #update operation

    GTot=copy.copy(G2D)
    GTot.operation=lambda *args: Goperation2D(*args)+GoperationSubs(*args)

    #--- Kill the propogating waves from the boundary, and just look at signal vs. plasmon wavelength
    Q=20
    sigma2D.set_value(1-1j/Q)
    sigmaPML.set_value(-20j)
    beta.set_value(.5) #for hBN and everything else at freq=900 cm-1

    x0=10
    wls_screened=np.linspace(1,30,30)
    wls=wls_screened/(1-beta)
    RTot=spectroscopy(GTot,WL,wls,x0=x0,excitation=excitation,\
                       a=1,param_name='$\lambda_p$')

    """
        charge2D = (sigma2D*Graphene + sigmaPML*PML) #total laplacian operator for conductivities
        G2D=S.GeneralOperator((beta,V,charge2D,WL),basis=Sample,\
                operation=lambda beta,V,charge2D,WL: Goperation2D(beta,V,charge2D,WL,subs=False)) #Our Green's function
        G2D_0=S.GeneralOperator((beta_bare,V,charge2D,WL),basis=Graphene,\
                operation=lambda beta,V,charge2D,WL: Goperation2D(beta,V,charge2D,WL,subs=False)) #Our Green's function

        GSubs=copy.copy(G2D) #Same operators, same basis, so just copy
        GSubs.operation=lambda beta,V,charge2D,WL: Goperation2D(beta,V,charge2D,WL,subs=True) #update operation

        GTot=copy.copy(G2D)
        GTot.operation=lambda *args: Goperation2D(*args,subs=False)+GoperationSubs(*args,subs=True)
    """

def build_PML(dx,X,Y,L,sigmaPML,Sample):
    w_PML_frame = 2*dx #width of PML issues
    PML_frame1=1/((X-L/2)**2+w_PML_frame**2)+1/((X+L/2)**2+w_PML_frame**2)
    PML_frame2=1/((Y-L/2)**2+w_PML_frame**2)+1/((Y+L/2)**2+w_PML_frame**2)
    PML_frame=w_PML_frame**2*np.where(PML_frame1>PML_frame2,PML_frame1,PML_frame2)

    #--- Boundary extending all the way to edge has numerical issues
    buf=dx
    PML_profile=PML_frame*((np.abs(X)<(L/2-buf))*(np.abs(Y)<(L/2-buf)))

    #--- set strictly equal to zero inside some box, then level the remaining PML
    w_PML=40*dx
    PML_domain=((np.abs(X)>(L/2-w_PML)) \
               +(np.abs(Y)>(L/2-w_PML))) #a frame
    PML_profile*=PML_domain
    PML_profile[PML_profile>0]-=PML_profile[PML_profile>0].min()

    plt.figure()
    AWA(1j*sigmaPML*PML_profile,adopt_axes_from=Sample.AWA).real.plot()
    plt.title('$\sigma_{PML}$')

    plt.figure()
    AWA(1j*sigmaPML*PML_profile,adopt_axes_from=Sample.AWA).real.cslice[:,0].plot()
    plt.title('$\sigma_{PML}$ cross section')
    plt.show()

    return S.GridLaplacian_aperiodic(dx,sigma=PML_profile)

#--- This is just a dipole, emulating the field from a tip
def excitation(Rx,Ry,a=1):
    direction=[0,1]
    z=5*a #This gives a qpeak=1/(2*z)=1/(10a) (Jiang & Fogler) and lambda_peak=2*pi*10a
    r=np.sqrt(Rx**2+Ry**2+z**2)
    rho=np.sqrt(Rx**2+Ry**2)
    rhat_rho=rho/r
    rhat_z=z/r

    exc=(direction[0]*rhat_rho+direction[1]*rhat_z)/r**2

    return AWA(exc,adopt_axes_from=Sample.AWA)

def spectroscopy(G,param,vals,param_name=None,\
                 excitation=excitation,x0=10,y0=0,\
                 verbose=True,**kwargs):

    if not param_name: param_name='parameter'

    if verbose: print('Staging excitation into basis...')

    #--- Make excitation
    X,Y=G.XY
    exc=S.normalize(excitation(X-x0,Y-y0,**kwargs))
    vE=G.vectors_from(exc)

    #--- Define how we will collect the response: project back into excitation
    collect_in=(~PML_domain.astype(bool)) #collect in region outside PML
    vC=G.vectors_from(exc*collect_in)

    spectrum=[]
    for val in vals:
        if verbose: print('Working on `%s=%1.2f`'%(param_name,val))
        param.set_value(val)
        R=G.matrix
        #--- Get response at each value of `param`
        spectrum.append(np.complex(vC.T @ R @ vE))

    return AWA(spectrum,axes=[vals],axis_names=[param_name])

#--- Define Green's functions
# This is derived by writing down non-local Helmholtz equation with substrate screening
# This is the real machinery

def Goperation2D(beta,V,sigma2D,Graphene,sigmaPML,PML,WL):
    Vscr = (1-beta)*V #screened Coulomb interaction
    charge2D = (sigma2D*Graphene + sigmaPML*PML) #total laplacian operator for conductivities

    r2D = Vscr/(2*np.pi)*charge2D*1/(Vscr/(2*np.pi)*charge2D - 2*np.pi/WL) #screened response of graphene

    return r2D #+rS

def Goperation2D_0(beta,V,sigma2D,Graphene,sigmaPML,PML,WL):
    Vscr=V
    charge2D = (sigma2D*Graphene + sigmaPML*PML) #total laplacian operator for conductivities

    r2D = Vscr/(2*np.pi)*charge2D*1/(Vscr/(2*np.pi)*charge2D - 2*np.pi/WL) #screened response of graphene

    return r2D #+rS

def GoperationSubs(beta,V,sigma2D,Graphene,sigmaPML,PML,WL):

    Vscr = (1-beta)*V #screened Coulomb interaction
    charge2D = (sigma2D*Graphene + sigmaPML*PML) #total laplacian operator for conductivities

    rS = beta*(-2*np.pi/WL)*1/(Vscr/(2*np.pi)*charge2D - 2*np.pi/WL) #screened response of substrate

    return rS #r2D#+rS

"""
def Goperation2D(beta,V,charge2D,WL,subs=False):
    Vscr = (1-beta)*V #screened Coulomb interaction
    if subs:
        r2D = beta*(-2*np.pi/WL)*1/(Vscr/(2*np.pi)*charge2D - 2*np.pi/WL) #screened response of substrate
    else:
        r2D = Vscr/(2*np.pi)*charge2D*1/(Vscr/(2*np.pi)*charge2D - 2*np.pi/WL) #screened response of graphene

    return r2D
"""

if __name__=="__main__": SDai_hBN_results()
