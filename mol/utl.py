from __future__ import division

import numpy as np
import scipy as sp
from mol.mol import CONSTANTS, van_der_waals_radii

'''
Accelerated Broadband Spectra Using Transition Dipole Decomposition and Pade Approximants by Bruner, LaMaster and Lopata -
[https://onesixtwo.club/scv/src/1584661408095.pdf]
Real-time time-dependent electronic structure theory by Joshua J.Goings, Patrick J.Lestrange and Xiaosong Li -
[https://joshuagoings.com/assets/real-time-electronic-structure.pdf]
Toeplitz - [https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html]
Thomson problem -[https://en.wikipedia.org/wiki/Thomson_problem]
'''

def pade(a, b, w):
    #Compute Pade approximant via extended Euclidean algorithm for
    #polynomial greatest common divisor - ref[1] equations (28)(29)

    return np.poly1d(a)(w)/np.poly1d(b)(w)

def solve_pade(x, y, damping, points, interval, tick=0.0001):
    #solve for spectrum, ref [1] equations (30)-(35)

    #zero at t=0, y[0] = 0 !
    y -= y[0]

    #apply damping - Lorentzian shape
    step = x[1] - x[0]
    damping = np.exp(-(step * np.arange(len(y))) / float(damping))
    y *= damping

    #diagonal Pade scheme
    n = min(len(y)//2, points)

    #generate vector and limit points
    X = -y[n+1:2*n]

    #compute Toeplitz matrix [n-1, n-1] and solve
    A = sp.linalg.toeplitz(y[n : 2*n-1], r=y[n : 1: -1])
    try:
        b = np.linalg.solve(A, X)
    except:
        exit('singular matrix - no field')

    #[1, [n-1]] -> [n] column vector
    b = np.hstack((1.0, b))

    #v[n]*toeplitz[n,n] a strictly lower triangular matrix
    a = np.einsum('pr,r->p', np.tril(sp.linalg.toeplitz(y[0 : n])),b, optimize=True)

    #frequency range
    frequency = np.arange(0.0 , interval , tick)

    w = np.exp(-1j*frequency*step)

    #Pade approximant via extended Euclidean algorithm
    fw = pade(a, b, w)

    return fw, frequency

def get_spectrum(time, dipole, damping=50.0, points=5000, interval=2.0, tick=0.0001, field = [None, 'i', 0.001]):
    #solve from spectra

    #generate field profile from shaape and time vector
    pulse, field_type, strength = field
    field_signal = [pulse(t) for t in time]

    dipole_spectrum , frequency = solve_pade(time, dipole, damping, points, interval, tick)
    field_spectrum, _           = solve_pade(time, field_signal, damping, points, interval, tick)

    #get return spectrum type real, imaginary or absolute
    omega = dipole_spectrum.real if field_type == 'r' else dipole_spectrum.imag if field_type == 'i' else np.abs(dipole_spectrum)

    #absorption formula
    spectrum = (4.0*CONSTANTS('hartree->eV')*CONSTANTS('alpha')*interval*frequency*np.pi*omega)/(3.0*np.abs(field_spectrum))

    return frequency, spectrum

def peaks(spectrum, frequency, tolerance):
    #find the peaks in the spectrum aove tolerance

    from scipy.signal import argrelmax as pks

    extrema = pks(np.abs(spectrum))

    #apply tolerance
    idx = np.where((np.abs(spectrum[extrema]) >= tolerance))
    jdx = extrema[0][idx[0]]

    nPeaks = len(jdx)
    peaks = np.zeros(nPeaks)
    for i in range(nPeaks):
        peaks[i] = frequency[jdx][i]

    return peaks

def get_atomic_weight(z, unit='amu'):
    #values from NIST

    weight = np.array([0.0,  1.00784,     4.002602,
                             6.938,       9.0121831, 10.806,     12.0096, 14.00643,     15.99903, 18.998403163, 20.1797,
                             22.98976928, 24.304,    26.9815384, 28.084,  30.973761998, 32.059,   35.446,       39.948])

    if unit == 'amu': weight *= CONSTANTS('em2->amu')

    return weight[z]

def integrator(scf, id, dt, cache, method='velocity-verlet'):
    #velocity-Verlet, Beeman or Adams-Moulton integrators

    if   method == 'velocity-verlet':
        return 0.5 * dt * (cache[-2][id, :] + cache[-1][id, :]) / get_atomic_weight(scf.mol.atom[id].number)
    elif method == 'beeman':
        return dt * (2.0 * cache[-1][id, :] + 5.0 * cache[-2][id, :] - cache[-3][id, :]) / (get_atomic_weight(scf.mol.atom[id].number) * 6.0)
    elif method == 'adams-moulton':
        return dt * (5.0 * cache[-1][id, :] + 8.0 * cache[-2][id, :] - cache[-3][id, :]) / (get_atomic_weight(scf.mol.atom[id].number) * 12.0)

def is_bond(scf, p, q):
    #determine if there is a bond between atoms

    if p.id == q.id: return False
    tolerances = {'vdw': 1.6, 'mayer': 0.7}

    #by Van der Walls radii
    vdw_seperation = ((van_der_waals_radii[p.symbol] + van_der_waals_radii[q.symbol]) /
                     CONSTANTS('bohr->angstrom'))

    has_bond_vdw = (np.linalg.norm(p.center[:] - q.center[:]) < tolerances['vdw'] * vdw_seperation)

    #by Mayer bond-order
    mayer = 2.0 * np.einsum('pr,rq->pq', scf.get('d'), scf.get('s'), optimize=True)

    bond_order = np.zeros((scf.mol.natm, scf.mol.natm))

    for i, x in enumerate(scf.mol.orbital):
        for j in range(i):

            mu = x.atom.id
            nu = scf.mol.orbital[j].atom.id

            if mu != nu: bond_order[mu, nu] += mayer[i, j] * mayer[j, i]

    bond_order += bond_order.transpose()
    has_bond_mayer = bond_order[p.id, q.id] > tolerances['mayer']

    return has_bond_vdw and has_bond_mayer

def thomson_distribution(n, tolerance=1e-6, cycles=10000, view=None):
    '''
    Thomson problem of distributing electrons on surface of sphere under electrostatic
    forces
    '''

    def visualize(surface, azimuth=30, elevation=30):
        import pylab as p
        fig = p.figure()
        ax = fig.add_subplot(111, projection='3d')


        ax.scatter3D(surface[:,0], surface[:,1], surface[:,2])

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo['grid']['color'] = (0, 0, 0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0

        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(azim=45, elev=45)
        p.show()

    import random
    import numpy as np

    tolerance = 1e-6

    surface = np.zeros((n, 3))
    for p in range(n):

        #random conforming to probability density function
        theta         = random.random() * 2*np.pi
        phi           = np.arcsin(random.random() * 2 - 1)
        surface[p, :] = [np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]

    #perform the optimization
    cycle = 1
    while True:

        #Total force
        forces = np.zeros((n, 3))
        for p in range(n):

            total_force = 0.0
            for q in range(n):
                if p == q: continue

                #distance vector, and it separation
                dv = surface[p] - surface[q]

                #Force vector representation [L]
                fv = dv / pow(np.linalg.norm(dv), 3)

                #Add to the total force at the point p
                forces[p] += fv

            #Total force over surface
            total_force += np.linalg.norm(forces[p])

        #Scale the forces to keep conservative
        force_scale = 0.25 / total_force if total_force > 0.25 else 1.0

        #displace each point and note distance displaced
        displaced_surface = surface + forces * force_scale
        normed_surface = np.array([displaced_surface[i]/np.linalg.norm(displaced_surface[i]) for i in range(n)])

        delta_surface = surface - normed_surface
        displacement = np.sum(np.linalg.norm(delta_surface))

        surface = normed_surface

        #Check for convergence and finish.
        if displacement < tolerance:
            break

        if cycle > cycles:
            return None, cycle, None

        cycle += 1

    #calculate UThom
    u_thomson = 0.0
    for i in range(n):
        for j in range(i+1, n):
            u_thomson += 1/(np.linalg.norm(surface[i, :] - surface[j,:]))

    if view is not None:
        visualize(surface, azimuth=view[0], elevation=view[1])

    return surface, cycle, u_thomson

class waveform(object):
    #supply a Gaussian or Lorentzian waveform function

    def __init__(self):
        pass

    @staticmethod
    def gaussian(domain, x, rho, broaden=True):
        '''
        return a Gaussian wavefunction. rho is the standard deviation. domain are the
        poles and for broadening x is the base linespace. Broaden = False gives the
        maximum amplitude for stick graph.
        '''

        f = 1/(rho * np.sqrt(np.pi * 2.0))
        if not broaden: return f

        return f * np.exp(-0.5*pow((domain - x)/ rho, 2.0))

    @staticmethod
    def lorentzian(domain, x, rho, broaden=True):
        '''
        return a Lorentzian wavefunction. rho is the full-width at half maximum. domain
        are the poles and for broadening x is the base linespace.
        '''

        f = 2.0 / (np.pi * rho)
        if not broaden: return f

        return (2.0 * rho / (np.pi)) / (pow(2*(domain - x), 2.0) + pow(rho, 2.0))
