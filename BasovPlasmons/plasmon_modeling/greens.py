from dolfin import *
from fenics import *
from mshr import *
import matplotlib.tri as tri

from scipy import special as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp


#       ------------------------------------------------------------------------------------
#                 _              _           _            _            _             _
#                /\ \           /\ \        /\ \         /\ \         /\ \     _    / /\
#               /  \ \         /  \ \      /  \ \       /  \ \       /  \ \   /\_\ / /  \
#              / /\ \_\       / /\ \ \    / /\ \ \     / /\ \ \     / /\ \ \_/ / // / /\ \__
#             / / /\/_/      / / /\ \_\  / / /\ \_\   / / /\ \_\   / / /\ \___/ // / /\ \___\
#            / / / ______   / / /_/ / / / /_/_ \/_/  / /_/_ \/_/  / / /  \/____/ \ \ \ \/___/
#           / / / /\_____\ / / /__\/ / / /____/\    / /____/\    / / /    / / /   \ \ \
#          / / /  \/____ // / /_____/ / /\____\/   / /\____\/   / / /    / / /_    \ \ \
#         / / /_____/ / // / /\ \ \  / / /______  / / /______  / / /    / / //_/\__/ / /
#        / / /______\/ // / /  \ \ \/ / /_______\/ / /_______\/ / /    / / / \ \/___/ /
#        \/___________/ \/_/    \_\/\/__________/\/__________/\/_/     \/_/   \_____\/
#       ------------------------------------------------------------------------------------

# Logs:
#     1.10.19 To fix, the direction of the scanning is chosen to be very specific, it would be interesting to see different scanning directions. Since it is assumed it makes it non-intuitive for the user
#     4.25.19 We notice that the process fenics function is sometimes flips the image on the y_axis. In order to fix this, one needs to reverse the input array on the y axis. (ACTIVE: Why does this happen?)

####### Functions for probing the Response Underneath the Tip #######

def scanning_behaviour(nx,ny,q=np.pi*2,density=1,x_lim=[-1,1],y_lim=[-1,1]):
    """
    Quickly creates the scanning behavior of launching a bessel function from the tip which will then couple to
    the eigenfunctions of the samples in order to create a greens function. Implements this by making a matrix
    with double the required dimensions and taking submatrices rather than redoing the calculation. By choosing
    the window size nx, ny, we selected the range over which the scan occurs. An assumption about the density of the
    result is made. Default nx and ny should both be set to 100.

                nx
        ------------------
        |           (    |
        |          (    (|
        |         (    ( |
        |        (    (  |
        |       (    (   |  ny
        |      (    (    |
        |     (    (   ( |
        |    (    (   (  |
        |   (    (   (   |
        ------------------

    Args:
        param1: nx (int) x window size in terms of entries
        param2: ny (int) y window size in terms of
        param3: q (float) momentum of the bessel function
        param4: density (float) will act like a zoom function, default set to 1

    Returns:
        Numpy array with the shape ((nx+1)*(ny+1),nx+1,ny+1)
    """
    #We are just going to use this range
    x_multiplier = (nx/100)*density
    y_multiplier = (ny/100)*density
#     n = max(nx,ny)

    xs = np.linspace(2*x_lim[0]*x_multiplier,2*x_lim[1]*x_multiplier,2*nx+1)
    ys = np.linspace(2*y_lim[0]*y_multiplier,2*y_lim[1]*y_multiplier,2*ny+1)

    output = np.zeros((2*nx+1,2*ny+1))
    for i in range(len(xs)):
            for j in range(len(ys)):
                output[i][j] = q*np.sqrt(xs[i]*xs[i] + ys[j]*ys[j])

    big_array =sp.jv(1,output)

    scanning = []
    #This behavior scans from the bottom corner to the top
    for x in range(nx+1):
        for y in range(ny+1):
            scanning.append(big_array[x:x+nx+1,y:y+ny+1])

    return np.array(scanning)

def extract_SNOM_response(scanning, eigenvalues_eigenvector_dict,omega=5000+100j):
    """
    This function takes in scanning which consists of a scanning dimension and two spatial dimensions and a dictionary
    filled with eigenvalue eigenvector pairs are then used to construct a greens function at each point in the scanning
    dimension, then point where the SNOM tip would sense is extracted from the scanning dimension and an image of what
    the SNOM would see if created.

    Args:
        param1: scanning, a 3 dimensional matrix that consists of a scanning dimension and two spatial dimensions.
        param2: eigenvalue_eigenvector_dict dictionary filled with eigenvalue eigenvector pairs.
    Returns:
        2D image of what a SNOM tip would see.
    """
    #Taking the scanning matrix which consists of the scanning dimension and two spatial dimensions
    #and removing one of the spatial dimensions by "rolling it out"
    rolled_out = np.reshape(scanning, (scanning.shape[0],scanning.shape[1]*scanning.shape[2]))

    #Splitting the eigenvalue and eigenvectors into two arrays
    evalues, evecs = zip(*eigenvalues_eigenvector_dict.items())
    evecs = np.array(evecs)
    evalues = np.array(evalues)

    #unraveling
    evecs = np.reshape(evecs,(len(list(evecs)),evecs[0].shape[0]*evecs[0].shape[1]))

    #Step 1
    evecwexcite = np.dot(evecs,rolled_out.T)

    #diag
    denom = np.real(1/(evalues-omega))
    A = np.zeros((denom.shape[0],denom.shape[0]))+np.diag(denom,0)

    #Step 2, evals-lambda
    coef_matrix = np.dot(A,evecwexcite)

    #Step 3, back to evecs
    result = np.dot(coef_matrix.T,evecs)

    mod_result=np.reshape(result,(scanning.shape[0],scanning.shape[1],scanning.shape[2]))
    check = extract_excitation_response(scanning.shape[1]-1,scanning.shape[2]-1,mod_result)

    #Seems to need a correction, it comes out as the transpose of the proper image
    return check.T

def extract_excitation_response(nx,ny,excitation_response_matrix):
    """
    Method used to the response underneath the tip done algorithmically.

    Args:
        param1: nx, number of points sampled in the x dimension
        param2: nx, number of points sampled in the y dimension
        param3: excitation_response_matrix, matrix that contains a scanning dimension and two spatial dimensions
            that represent a greens function coupling a bessel function at a specific point.
    Returns:
        Image made up of points that represent what a SNOM tip would observe from the sample while scanning

    """
    output = np.zeros(excitation_response_matrix[0].shape)
    i=0
    for y in range(ny,-1,-1):
        for x in range(nx,-1,-1):
            output[x][y] = excitation_response_matrix[i][y][x]
            i+=1
    return output

def extract_excitation_matrix(excitation_response_matrix):
    """
    Method used to generate a matrix that can extract the response of underneath the tip through matrix multiplication. (Method was slower than the extract_excitation_response therefore we may refrain from using it)
    """
    output = np.zeros((excitation_response_matrix.shape[0],excitation_response_matrix[0].shape[0]*excitation_response_matrix[0].shape[1]))
    i=0
    for y in range(excitation_response_matrix[0].shape[1]-1,-1,-1):
        for x in range(excitation_response_matrix[0].shape[0]-1,-1,-1):
            output[i][excitation_response_matrix[0].shape[0]*y+x]=1
            i+=1
    return output

### Functions for Determining "Aiming Function"

def simple_bessel(shape,q=2*np.pi):
    #creating the bessel function
    bessel_function = np.zeros(shape)

    #coordinates and density of the coordinates for the creation of the bessel function
    xs = np.linspace(-1,1,shape[0])
    ys = np.linspace(-1,1,shape[1])

    for i in range(len(xs)):
        for j in range(len(ys)):
            bessel_function[i][j] = sp.jv(0,q*np.sqrt(xs[i]*xs[i]+ ys[j]*ys[j]))

    return bessel_function


def generate_greens_function_simpsons(excitation,eigenvalue_eigenfunction_dict,omega=5000+100j):
    """
    We take an excitation and eigenvalue eigenvector pairs in order to return the greens function fonp.reshape(final_result,(100,250))r the image.
    The excitations and eigenvector must be of the same dimensions in order to perform pairwise multiplication

    143 ms ± 675 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    #assert(exictation.shape)
    #storing the result
    entire_result = np.zeros((101,101))

    #determining which eigenfunctions had the largest impact
    dot_product_dictionary=dict()

    #Performing the 2D Integration
    for key, value in eigenvalue_eigenfunction_dict.items():
        x_arb= np.linspace(-1,1,101)
        y_arb= np.linspace(-1,1,101)

        #Absolute value taken just to examine impact
        dot_product_dictionary[key]=abs(np.sum(excitation*value))
        entire_result+=(simps(simps(excitation*value,y_arb),x_arb)*value)*np.real(1/(key-omega))

    return (entire_result, dot_product_dictionary)

def generate_greens_function_vectorized(excitation,eigenvalue_eigenfunction_dict,omega=400+100j):
    """
    We take an excitation and eigenvalue eigenvector pairs in order to return the greens function for the image.
    The excitations and eigenvector must be of the same dimensions in order to perform pointwise multiplication.
    Returns the resulting greens function and a dot_product_dict which contains information on the eigenvalue-
    eigenvector pairs that had the most significant influence on the formation of the greens function. Uses a
    vectorized implementation that runs in 36.6 ms.

    Speed: 36.6 ms ± 376 µs

    Args:
        param1: excitation, numpy array that typically will contain a bessel function
        param2: eigenvalue_eigenfunction_dict, dictionary that contains eigenvalue eigenvector pairs
        param3: omega, energy we are exciting the system at (***)
    Returns:
        result, dot_product_dict where result contains the image fo the greens function itself and dot_product_dict
        contains eigenvalue with eigenfunction-excitation dot product pairs to determine where to 'aim' the function

    """
    entire_result = np.zeros(excitation.shape)

    excite = np.reshape(excitation,(excitation.shape[0]*excitation.shape[1]))

    evalues, evecs = zip(*eigenvalue_eigenfunction_dict.items())
    evecs = np.array(evecs)
    evalues = np.array(evalues)

    #pointwise multiplication with the excitations and the eigenvector
    evecs = np.reshape(evecs,(len(list(evecs)),evecs[0].shape[0]*evecs[0].shape[1]))
    evecwexcite = np.dot(evecs,excite.T)

    #(evalues,evecwexcite) contains all the information for aiming
    dot_product_dict = dict(zip(evalues, evecwexcite))

    #performing scaling with respect to 1/omega-omega_0
    denom = np.real(1/(evalues-omega))
    A = np.diag(denom,0)

    #Step 2, evals-omega
    coef_matrix = np.dot(A,evecwexcite)

    #Step 3, back to evecs
    result = np.dot(coef_matrix.T,evecs)
    result = np.reshape(result,excitation.shape)

    return result,dot_product_dict

def generate_aiming_function(q_dict, processed_eigenvalue_eigenfunction_pairs, gen_func=generate_greens_function_vectorized, omega=5000+100j):
    """
    This function is meant to determine which eigenvalue has the most significant impact, dot product, with the
    excitation. We can then tune omega so that we can achieve the singularity from 1/(omega-omega_0) where
    omega=eigenvalue. Then our real term is simply the term
    """
    q_value_dictionary = dict()
    for q, excitation in q_dict.items():
        result, dot_product_dict = gen_func(excitation,processed_eigenvalue_eigenfunction_pairs,omega=omega)
        dot, mag = zip(*dot_product_dict.items())
        #plt.figure();plt.plot(dot,mag)
        maximum=max(dot_product_dict.values())
        result = filter(lambda x:x[1] == maximum,dot_product_dict.items())
        q_value_dictionary[q]=list(result)[0][0]

    qs, values = zip(*q_value_dictionary.items())

    #All Data Plotted
    plt.figure();plt.plot(qs, values)
    plt.xlabel("q")
    plt.ylabel("Eigenvalues")
    plt.title("Eigenvalues of the Eigenfunction w/ Most DP w.r.t q")

    #Determining the Coefficients

    coef = np.polyfit(qs, values,2)
    g = lambda x,y: np.dot(y,np.array([x**(len(y)-i-1) for i in range(len(y))]))

    x = np.linspace(0,100,1000)
    plt.plot(x, g(x,coef),c='r')
    plt.plot(qs, values)
    plt.ylabel("Most Influential Eigenvalue")
    plt.xlabel("q")
    plt.title("Aiming Coefficients with omega = {}".format(omega))
    return lambda x: g(x,coef)

def omega_to_complex(omega,Q=1):
    return complex(omega,omega/Q)
