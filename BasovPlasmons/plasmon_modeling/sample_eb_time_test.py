import sys,h5py,time
from dolfin import *
from fenics import *
from mshr import *
import Plasmon_Modeling as PM
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
from scipy import special as sp
import numpy as np

sigma = PM.S()

#Using lambda and L
sigma.set_sigma_values(10,10)
s_1,s_2 = sigma.get_sigma_values()

#We are simply using the geometry of the space in order to get results.
#Therefore we do not need a RectangularSample object.

#Using a simple square mesh along with the helholtz solver in order to get eigenfunction eigenvalue pairs
mesh = UnitSquareMesh(200, 200)
start = time.time()
eigenvalue_eigenfunction_pairs = PM.helmholtz(mesh, 5, number_extracted=1000,sigma_2 = s_2,to_plot=True)
print('Time elapsed: {} seconds'.format(time.time()-start))
