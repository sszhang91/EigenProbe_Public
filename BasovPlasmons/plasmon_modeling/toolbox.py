from dolfin import *
from fenics import *
from mshr import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
import random
import numpy as np

#------------------------------------------------Plotting and Extraction Tools-----------------------------------------------------#

def mesh2triang(mesh):
    """
    Helper Method for mplot, process_fenics_eigenfunction, etc.
    """
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def process_fenics_function(mesh,fenics_function, x_axis=np.linspace(0,1,100), y_axis=np.linspace(0,1,100)\
                                 ,to_plot=True):
    """
    Method in order to convert fenics functions into np arrays, in order to be able to deal with objects of arbitrary dimension
    the outputs will be masked Fenics Functions. This will only apply to 2D functions that need to be interpolated.
    This is build on top of the matplotlib.tri module and is effective at interpolating from non-uniform meshs.
    Args:
        param1: mesh that the fenics function is defined on
        param2: fenics function that needs to be converted into a
        param3: x_axis (np.array) needs to be a 1-D array which represents the units of the x-axis
        param4: y_axis (np.array) needs to be a 1-D array which represents the units of the y-axis
        param5: to_plot (bool) determines if the image of the interpolated function should be plotted
    Returns:
        Masked Numpy Array with the required eigenfunction
    """
    V = FunctionSpace(mesh, 'CG', 1)
    fenics_function = interpolate(fenics_function, V)

    C = fenics_function.compute_vertex_values(mesh)

    yv, xv = np.meshgrid(y_axis, x_axis) #@ASM 2020.05.16: `np.meshgrid` is silly in that column-changing array is returned first

    test = tri.LinearTriInterpolator(mesh2triang(mesh),C)(xv,yv)

    if to_plot:
        plt.imshow(test,cmap='seismic')

    return test

def interpolate_fenics_eigpairs(fenics_eigpairs,\
                                xlims,ylims,Nx=200,\
                                Neig=1000,\
                                eigval_minus_one=True,\
                                verbose=True,adjust_x=0,adjust_y=0):
    """Interpolate a dictionary of fenics functions to a regular
    grid defined by `xlims`, `ylims`, and `Nx` pixels along the
    x-direction.  As many as `Neig` functions will be interpolated.
    To subtract one from eigenvalue keys in `fenics_eigpairs`,
    use the `eigval_minus_one` flag.  (The necessity of this
    flag remains unknown.)"""
    
    import time
    import sys
    from common.baseclasses import AWA
    
    f=list(fenics_eigpairs.values())[0]
    mesh=f.function_space().mesh()
    
    Lx=np.abs(xlims[1]-xlims[0]); dx=Lx/Nx
    Ly=np.abs(ylims[1]-ylims[0]); Ny=int(round(Ly/dx))
    
    #We use `xmin+(Nx-1)*dx` so that we don't double-count the boundary
    Xs,Ys=np.ogrid[xlims[0]-adjust_x:xlims[0]-adjust_x+(Nx-1)*dx:Nx*1j,\
                   ylims[0]-adjust_y:ylims[0]-adjust_y+(Ny-1)*dx:Ny*1j]
    xs,ys=Xs.squeeze(),Ys.squeeze()
    
    Neig=min((Neig,len(fenics_eigpairs)))
    
    i=0
    eigpairs={}
    t0=time.time()
    for eigval, eigfunc in fenics_eigpairs.items():
        if verbose:
            sys.stdout.flush()
            sys.stdout.write('Progress: %1.2f%%\r'%(i/Neig*100))
        interpolated=process_fenics_function(mesh,eigfunc, x_axis=xs, y_axis=ys,to_plot=False)
        interpolated[np.isnan(interpolated)]=0
        if eigval_minus_one: eigval-=1
        eigpairs[eigval]=AWA(interpolated,axes=[xs,ys]) #Still don't understand why `-1` is needed
        i+=1
        if i>=Neig: break
    
    if verbose: print('Processing time per function: %s'%((time.time()-t0)/i))
    
    return eigpairs

def mplot(obj):
    """Plots fenics functions in matplotlib.pyplot
        Args:
            param1: FEniCS function
        Returns:
            None
    """
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C,cmap='seismic')
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud',cmap='seismic')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')

def plot_fenics(z):
    fig = plt.figure()
    #         plt.subplot(131);mplot(mesh);plt.title("Mesh"),plt.tick_params(
    #             axis='both',          # changes apply to the x-axis
    #             which='both',      # both major and minor ticks are affected
    #             bottom=False,
    #             right = False,     # ticks along the bottom edge are off
    #             left = False,
    #             top=False)         # ticks along the top edge are off)
    real = z.split(deepcopy=True)[0]
    r_lim = max(abs(max(real.vector())),abs(min(real.vector())))
    plt.subplot(121);mplot(real);plt.title("Real Part"),plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        right = False,     # ticks along the bottom edge are off
        left = False,
        top=False,         # ticks along the top edge are off
        labelleft=False);
    #cax = make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05);plt.colorbar(cax=cax);plt.clim(-r_lim,r_lim)


    imaginary = z.split(deepcopy=True)[1]
    #i_lim = max(abs(max(imaginary.vector())),abs(min(imaginary.vector())))
    plt.subplot(122);mplot(imaginary);plt.title("Im Part"),plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        right = False,     # ticks along the bottom edge are off
        left = False,
        top=False,         # ticks along the top edge are off
        labelleft=False);
    #cax = make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05);#plt.colorbar(cax=cax);plt.clim(-i_lim,i_lim)

    #         fig.subplots_adjust(right=0.8)
    #         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #         fig.colorbar(plt.subplot(133), cax=cbar_ax)
    plt.show()

##################### FUNCTIONS FOR ROTATING OBJECTS ##########################
#--------------------------------------------------------------------------------------------------------------------------------#

def calculated_centroid(vertices):
    """
    Function meant for calculating the centroid of a 2D polygon

    Args:
        vertices (list): list of pairs of vertices

    Returns:
        centroids (list): [x_centroid, y_centroid]
    """
    x_value, y_value = 0,0
    for i in vertices:
        x_value += i[0]
        y_value += i[1]

    return [x_value/len(vertices),y_value/len(vertices)]

def rotate_object(vertices_list, rotation_degrees):
    """
    Function meant for rotating a 2D polygon

    Args:
        vertices (list): list of pairs of vertices
        rotation_degrees (float): degree amount to rotate the vertices by

    Returns:
        output_vertices (list): list of pairs of vertices after being rotated
    """
    output_vertices = []
    theta = np.radians(rotation_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    for vertex in vertices_list:
        output_vertices.append([vertex[0]*c-vertex[1]*s,vertex[1]*c+vertex[0]*s])
    return output_vertices

def center_then_rotate(vertices, rotation_degrees):
    """
    Function to first center then rotate the vertices then transform it back. Otherwise the rotation
    would be around the origin.

    Args:
        vertices (list): list of pairs of vertices
        rotation_degrees (float): degree amount to rotate the vertices by

    Return:
        modified_vertices (list): list of pairs of vertices after being centered, rotated, then transformed back
    """
    #First calculate the centroid
    centroid = np.array(calculated_centroid(vertices))
    #subtract centroid from the points
    modified_vertices = [np.array(i)-centroid for i in vertices]
    modified_vertices=rotate_object(modified_vertices,rotation_degrees)
    modified_vertices = [np.array(i)+centroid for i in modified_vertices]

    return modified_vertices

def counter_clockwise_sort(vertices):
    """
    Function to remedy the error given when Fenics needs the vertices in counterclockwise order

    Args:
        vertices (list): list of pairs of vertices

    Return:
        output (list): list of pairs of vertices after being centered, rotated, then transformed back
    """
    #This first line looks suspicious...
    centroid=calculated_centroid(rotate_object(vertices,1))
    def func(array):
        print(array)
        return np.arctan(array[1]/array[0])

    to_sort=dict(zip(map(func,[np.array(i)-centroid for i in vertices]),vertices))
    # print(len(map(func,[np.array(i)-centroid for i in vertices])))
    # print(len(vertices))
    output=[]
    for key in sorted(to_sort):
        output.append(to_sort[key])

    return output