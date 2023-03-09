from mol.mol import molecule
from scf.rhf import RHF
import int.mo_spin as mos
from phf.eig import solver
from mol.mol import CONSTANTS
from int.aello import aello

import phf.cit as cit

import numpy as np
import scipy as sp

from phf.tdhf import TDHF, RT_TDHF

if __name__ == '__main__':
    #we're using the geometry from the Crawford Projects
    mol = molecule([['H', ( 0.000000000000,  0.000000000000 ,0.000000000000)], 
                    ['H', ( 0.000000000000,  0.000000000000 ,0.740000000000)]], 
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='3-21g',
                    silent=False)

    scf = RHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    '''
    First we get an analytic solution for the hydrogen molecule from time-dependent Hartree-Fock
    in the Tamm-Dancoff Approximation (CIS)
    '''

    td = TDHF(scf)  
    td.format(methods= 'energy, electric length, electric velocity') 

    '''
    define a function for the pulse to be applied as an external field, we'll use a narrow Gaussian
    but a kick is also shown (instantaneous pulse)
    '''

    def gaussian(t):
        #Gaussian waveshape

        rho =0.1
        field_strength = 0.0001

        return np.exp(-(t**2)/ (2 * rho * rho)) * field_strength

    def kick(t):
        #instantaneuos pulse

        field_strength = 0.0001
        return 0.0 if t != 0 else field_strength

    '''
    see if real-time TDHF can reproduce analytic results. Use the Magnus second order to time propogate
    the Fock matrix. We see that the electric length gauge dipole has only in the z-axis direction so 
    we will investigate in that axis.
    '''

    rt = RT_TDHF(scf, pulse=gaussian, dt=0.05, cycles=1000, axis='z')
    rt.execute('magnus')


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    plt.grid()
    ax[0].set_title('$H_2$ z-axis electric dipole')
    ax[0].set(xlabel='time [au]', ylabel='dipole (debye)')
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    rt.cache[:, 2] *= CONSTANTS('au->debye')
    ax[0].plot(rt.cache[:,0], rt.cache[:,2] ,'.k',label='Magnus 2nd order', markeredgecolor='none')
    ax[0].legend(loc=1)

    '''
    Generate spectrum
    '''
    from mol.utl import get_spectrum, peaks
    frequency, spectrum = get_spectrum(rt.cache[:,0], rt.cache[:,2], damping=50.0, points= 5000, interval=2.0, tick=0.0001, field=[gaussian, 'i', 0.0001])
    frequency *= CONSTANTS('hartree->eV')

    ax[1].set(xlabel='energy (eV)', ylabel='absorption (arb.)')
    ax[1].plot(frequency, spectrum)

    plt.tight_layout()
    plt.show()

    '''
    by inspection we see that the peaks occur at about 16eV and 47eV which look right. There is a *peaks* method in mol.utl which can estimate the position
    of the peaks more accurately
    '''
    peak_eV = peaks(spectrum, frequency, 0.5)
    print('\nReal-time peaks occur at  {:>6.2f}eV   and  {:>6.2f}eV'.format(peak_eV[0], peak_eV[1]))
    '''
    Better than 1%
    '''

'''    
******************
*   scf output   *
******************
method                  RHF 
charge                  0 
spin                    0 
units                   angstrom
open shell              False      :  multiplicity  1

basis is                3-21g
analytic integration    aello cython - McMurchie-Davidson scheme

diis                    True  : buffer size is  6
scf control             maximum cycles are  50    :  convergence tolerance  1e-10

basis analysis
--------------
shell   l    pGTO    cGTO
-------------------------
H    0 
  0     0      2       1
  1     0      1       1
H    1 
  2     0      1       1
  3     0      2       1

-------------------------
number of shells            4
number of primative GTO     6
number of contracted GTO    4

 cycle        E SCF              ΔE            |diis|               homo            lumo
------------------------------------------------------------------------------------------------
    1      -1.79209505       1.7921e+00      2.4886e-01         -0.54028862      0.29827182   
    2      -1.83199742       3.9902e-02      3.7828e-02         -0.58560366      0.26945395   
    3      -1.83718748       5.1901e-03      5.4818e-03         -0.59163756      0.26495255   
    4      -1.83792142       7.3394e-04      7.9036e-04         -0.59249440      0.26429730   
    5      -1.83802686       1.0544e-04      1.1387e-04         -0.59261758      0.26420276   
    6      -1.83608042       1.9464e-03      1.6405e-05         -0.59060510      0.26414723   
    7      -1.83803443       1.9540e-03      3.7960e-04         -0.59262782      0.26418637   
    8      -1.83804441       9.9841e-06      1.7475e-06         -0.59263811      0.26418683   
    9      -1.83804461       1.9253e-07      3.0841e-08         -0.59263831      0.26418684   
   10      -1.83804460       7.6398e-09      1.4749e-09         -0.59263830      0.26418684   
   11      -1.83804460       7.1098e-10      1.9672e-10         -0.59263830      0.26418684   
   12      -1.83804460       4.7383e-10      7.7798e-11         -0.59263830      0.26418684   
   13      -1.83804460       8.6069e-12      1.4442e-12         -0.59263830      0.26418684   
   14      -1.83804460       2.5757e-14      3.7981e-15         -0.59263830      0.26418684   

nuclear repulsion      0.7151043391
total electronic      -1.8380445979

final total energy    -1.1229402588

TDHF energy analysis
energy:      au          eV         nm       spin   spatial
------------------------------------------------------------
  1       0.579000    15.7554      78.69    2->4     1->2    
  2       1.180681    32.1280      38.59    2->6     1->3    
  3       1.713355    46.6228      26.59    2->8     1->4    

TDHF electric dipole length gauge analysis
           x        y       z           S      osc.
-----------------------------------------------------
  1     0.0000   0.0000   1.4381     2.0682   0.7983    
  2     0.0000   0.0000   0.0000     0.0000   0.0000    
  3     0.0000   0.0000   0.2501     0.0626   0.0714    

TDHF electric dipole velocity gauge analysis
           x        y       z           S      osc.
-----------------------------------------------------
  1     0.0000   0.0000  -0.6405     0.4102   0.4723    
  2     0.0000   0.0000   0.0000     0.0000   0.0000    
  3     0.0000   0.0000  -0.1924     0.0370   0.0144    

Real-time peaks occur at   15.51eV   and   46.39eV
'''