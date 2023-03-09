from mol.mol import molecule
from scf.rhf import RHF
import int.mo_spin as mos

import numpy as np

if __name__ == '__main__':
    mol = molecule([['O', (0.0, 0.0, 0.0)], 
                    ['H', (0,-0.757 ,0.587)], 
                    ['H', (0, 0.757 ,0.587)]], 
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
    
#**************************************
#* spin orbital energies denominators *
#**************************************
eps = mos.orbital_deltas(scf, 2)                               #returns a list of orbital differences upto and including level

#denominator at level 2 
eps_denominator = np.reciprocal(eps[1])

#*************************************
#* spin orbital 2-electron repulsion *
#*************************************
g = mos.orbital_transform(scf, 'm+s', scf.get('i'))

#**************************
#* MP2 energy computation *
#**************************

#get orbital slices 
nocc = sum(scf.mol.nele)                                       #occupied spin orbitals in number of alpha + beta electrons
o, v = slice(None, nocc), slice(nocc, None)                    #orbital occupation slices

td_amplitude = g[o, o, v, v]*eps_denominator
mp2_energy = 0.25 * np.einsum('ijab,ijab->', g[o, o, v, v], td_amplitude, optimize=True)
print('\nMoller-Plesset level 2 energy correction = {:>12.8f} Hartree'.format(mp2_energy))

#****************************
#* spin hartree-Fock energy *
#****************************

#get fock in spin orbitals
f = mos.orbital_transform(scf, 'm+s', scf.get('f'))            #get fock in molecular (m) and (+) spin (s) basis
e_spin_hf = np.einsum('ii', f[o, o]) - 0.5 * np.einsum('ijij', g[o, o, o, o]) + scf.mol.nuclear_repulsion()
print('SCF energy using spin orbitals           = {:>12.8f} Hartree'.format(e_spin_hf))
print('Spin-Spatial consistency check is       ', np.allclose(e_spin_hf, scf_energy))

print('\nFinal MP2 corrected energy               = {:<12.8f} Hartree'.format(e_spin_hf + mp2_energy))

#*********************************
#* Native Moller-Plesset methods *
#*********************************
from phf.mpt import MP

#MPn methods
print('\n*********************\n* MP native methods *\n*********************')
print('\nMP2 method                           {:<12.8f}'.format(MP(scf, 'MP2').correction))
assert np.isclose(MP(scf, 'MP2').correction, -0.122463321959357) #PySCF check
print('MP3 method                           {:<12.8f}'.format(MP(scf, 'MP3').correction))

#MP2 spin-summed
parallel, anti_parallel, spin_component_scaled = MP(scf, 'MP2r').correction
print('\nMP2      parallel  spin component    {:<12.8f}'.format(parallel))
print('MP2 anti-parallel  spin component    {:<12.8f}'.format(anti_parallel))
print('MP2 spin component sum               {:<12.8f}'.format(parallel + anti_parallel))
print('\nMP2 spin component scaled            {:<12.8f}'.format(spin_component_scaled))

#MP2 rdm and dipoles
hf, unrelaxed, relaxed = MP(scf, 'MP2mu').dipole
print('\nHF   reference dipole            {:<8.4f} {:<8.4f} {:<8.4f}     {:<8.4f} debyes'.
      format(hf[0], hf[1], hf[2], np.linalg.norm(hf)))
print('MP2  unrelaxed dipole            {:<8.4f} {:<8.4f} {:<8.4f}     {:<8.4f} debyes'.
      format(unrelaxed[0], unrelaxed[1], unrelaxed[2], np.linalg.norm(unrelaxed)))
print('MP2    relaxed dipole            {:<8.4f} {:<8.4f} {:<8.4f}     {:<8.4f} debyes'.
      format(relaxed[0], relaxed[1], relaxed[2], np.linalg.norm(relaxed)))

assert np.allclose([np.around(np.linalg.norm(hf),4), np.around(np.linalg.norm(unrelaxed),4), np.around(np.linalg.norm(relaxed),4)], [2.4366, 2.4175, 2.3637])

#orbital-optimized moller-plesset 2 OOMP2
scf.mol.silent = True
print('\nOrbital-optimized MP2 correction     {:<12.8f}'.format(MP(scf, 'OMP2').correction))

#Laplace transfom moller-plesset 2 LT-MP2
parallel, anti_parallel, mp2 = MP(scf, 'LT-MP2', parameter=120).correction
print('\nLaplace Transform MP2')
print('MP2      parallel  spin component    {:<12.8f}'.format(parallel))
print('MP2 anti-parallel  spin component    {:<12.8f}'.format(anti_parallel))
print('MP2 spin component sum               {:<12.8f}'.format(mp2))

#MP2 Natural orbitals
print('\nMP2 Natural Orbitals')
e, c = MP(scf, 'MP2rno').natural_mo
print('\u03A3 eigenvalues of MP2 odm {:<7.4f}'.format(sum(e)))

#check all eigenvectors are normal
assert np.allclose(np.array([np.linalg.norm(c[:, p]) for p in range(c.shape[1])]), np.ones(c.shape[1]))

#check eigenvectors are orthogonal
assert np.allclose(np.array([np.dot(c[:,0], c[:,p]) for p in range(1, c.shape[1])]), np.zeros(c.shape[1]-1))

# ******************
# *   scf output   *
# ******************
# method                  RHF 
# charge                  0 
# spin                    0 
# units                   angstrom
# open shell              False      :  multiplicity  1

# basis is                3-21g
# analytic integration    aello cython - McMurchie-Davidson scheme

# diis                    True  : buffer size is  6
# scf control             maximum cycles are  50    :  convergence tolerance  1e-10

# basis analysis
# --------------
# shell   l    pGTO    cGTO
# -------------------------
# O    0 
#   0     0      2       1
#   1     0      1       1
#   2     0      3       1
#   3     1      2       1
#   4     1      1       1
# H    1 
#   5     0      1       1
#   6     0      2       1
# H    2 
#   7     0      1       1
#   8     0      2       1

# -------------------------
# number of shells            9
# number of primative GTO    21
# number of contracted GTO   13

#  cycle        E SCF              ΔE            |diis|               homo            lumo
# ------------------------------------------------------------------------------------------------
#     1     -66.88334931       6.6883e+01      4.0114e+00         -0.14242248      0.82084771   
#     2     -97.30468743       3.0421e+01      2.6146e+00         -2.68127794     -0.25779453   
#     3     -76.29230421       2.1012e+01      2.6763e+00         -0.63680541      0.40411239   
#     4     -89.49819138       1.3206e+01      1.4815e+00         -1.81621132      0.08178714   
#     5     -82.21051725       7.2877e+00      9.4980e-01         -1.10461450      0.31772886   
#     6     -84.82412024       2.6136e+00      4.6645e-01         -1.33505193      0.26300553   
#     7     -84.76838889       5.5731e-02      1.3627e-02         -1.32875564      0.26370098   
#     8     -84.77394645       5.5576e-03      2.9864e-03         -1.32921159      0.26355369   
#     9     -84.77374235       2.0410e-04      2.2364e-04         -1.32924707      0.26354028   
#    10     -84.77366950       7.2845e-05      2.4951e-05         -1.32923749      0.26353274   
#    11     -84.77365997       9.5347e-06      4.1897e-06         -1.32923623      0.26353300   
#    12     -84.77365998       8.7719e-09      5.7775e-07         -1.32923618      0.26353301   
#    13     -84.77365973       2.5047e-07      9.5228e-08         -1.32923614      0.26353301   
#    14     -84.77365984       1.1900e-07      2.0716e-08         -1.32923614      0.26353301   
#    15     -84.77365980       4.3962e-08      9.6640e-09         -1.32923614      0.26353301   
#    16     -84.77365980       1.5251e-09      8.6170e-10         -1.32923614      0.26353301   
#    17     -84.77365980       2.2132e-10      2.1912e-10         -1.32923614      0.26353301   
#    18     -84.77365980       5.2268e-11      1.8664e-11         -1.32923614      0.26353301   
#    19     -84.77365980       6.0396e-12      2.1866e-12         -1.32923614      0.26353301   

# nuclear repulsion      9.1882584177
# total electronic     -84.7736598020

# final total energy   -75.5854013843

# Moller-Plesset level 2 energy correction =  -0.12246331 Hartree
# SCF energy using spin orbitals           = -75.58540138 Hartree
# Spin-Spatial consistency check is        True

# Final MP2 corrected energy               = -75.70786470 Hartree

# *********************
# * MP native methods *
# *********************

# MP2 method                           -0.12246331 
# MP3 method                           -0.00305142 

# MP2      parallel  spin component    -0.02780030 
# MP2 anti-parallel  spin component    -0.09466302 
# MP2 spin component sum               -0.12246331 

# MP2 spin component scaled            -0.12286239 

# HF   reference dipole            -0.0000  0.0000   2.4366       2.4366   debyes
# MP2  unrelaxed dipole            -0.0000  0.0000   2.4175       2.4175   debyes
# MP2    relaxed dipole            -0.0000  0.0000   2.3637       2.3637   debyes

# Orbital-optimized MP2 correction     -0.12328376 

# Laplace Transform MP2
# MP2      parallel  spin component    -0.02780030 
# MP2 anti-parallel  spin component    -0.09466287 
# MP2 spin component sum               -0.12246316 

# MP2 Natural Orbitals
# Σ eigenvalues of MP2 odm 10.0000
