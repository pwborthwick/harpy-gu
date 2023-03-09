from __future__ import division

from mol.mol import molecule
from scf.rhf import RHF

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

    scf.analyse('geometry,dipole,charge,bonds')

    #********************
    #* check with PySCF *
    #********************
    import numpy as np
    print(np.isclose(scf_energy, -75.5854013693264))

    #using cached values
    core_hamiltonian = scf.get('t') + scf.get('v')
    eri = scf.get('i') ; d = scf.get('d')
    f = core_hamiltonian + 2.0 * np.einsum('rs,pqrs->pq', d, eri, optimize=True) -  np.einsum('rs,prqs->pq', d, eri, optimize=True)

    assert np.allclose(f, scf.get('f'))

    mo_coeff = scf.get('c') ; mo_occ = scf.get('o')
    density =  np.einsum('pr,qr->pq', mo_coeff*mo_occ, mo_coeff, optimize=True)

    assert np.allclose(density, d)

    homo = scf.get('e')[int(sum(mo_occ))-1]
    assert homo, -0.3912742463712855

    assert np.trace(scf.get('s')), scf.mol.natm

    atom_number = 1
    atom = scf.mol.atom[atom_number]
    # print('atom ID {:2}   SYMBOL {:2s}   Z {:2} '.format(atom.id, atom.symbol, atom.number), atom.center)

    basis_number = 3
    basis = scf.mol.orbital[basis_number]
    # print('basis ATOM CENTER {:2}  SYMBOL {:4s}   PRIMATIVES {:2} '.format(basis.atom.id, basis.symbol, basis.primatives))
    # print('',basis.momenta,'\n',basis.exponents,'\n',basis.coefficients,'\n',basis.atom.center,'\n',basis.normals)

#*******************************************************************************************************************************
#                                                            OUTPUT                                                            *
#*******************************************************************************************************************************
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

# Geometry
# -----------
#  O       0.0000     0.0000     0.0000
#  H       0.0000    -1.4305     1.1093
#  H       0.0000     1.4305     1.1093

# dipole momemts (Debye)
# ----------------------
#  x= -0.0000 y=  0.0000 z=  2.4365         resultant   2.4365

# Lowdin populations
# --------------------
#   [8.4638 0.7681 0.7681]

# charge
#   [-0.4638  0.2319  0.2319]        net = -0.00

# Mayer bond orders
# ------------------
#        H1        H2
# O0 | 0.8308    0.8308
# H1 |           0.0052

# Valency
# O0  = 1.66:   H1  = 0.84:   H2  = 0.84:
# True
