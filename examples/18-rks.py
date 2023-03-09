from __future__ import division
import numpy as np

from mol.mol import molecule, van_der_waals_radii
from scf.rks import RKS


if __name__ == '__main__':

    geo = ([['O', ( 0.000000000000, -0.143225816552, 0.000000000000)],
            ['H', ( 1.638036840407,  1.136548822547, 0.000000000000)],
            ['H', (-1.638036840407,  1.136548822547, 0.000000000000)]])

    # Computation options
    mol = molecule( geo,
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RKS(mol,
              xc='LDA,VWN_RPA',
              mesh='fine',
              cycles=50,
              tol=1e-7,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    scf.analyse('geometry,dipole,charge,bonds')

    #********************
    #* check with PySCF *
    #********************
    import numpy as np
    print(np.isclose(scf_energy, -74.92899096451207))

    #using cached values - 'x' is exchange-correlation potential
    core_hamiltonian = scf.get('t') + scf.get('v')
    eri = scf.get('i') ; d = scf.get('d')
    f = core_hamiltonian + 2.0 * np.einsum('rs,pqrs->pq', d, eri, optimize=True) + scf.get('x')

    assert np.allclose(f, scf.get('f'))

    mo_coeff = scf.get('c') ; mo_occ = scf.get('o')
    density =  np.einsum('pr,qr->pq', mo_coeff*mo_occ, mo_coeff, optimize=True)

    assert np.allclose(density, d)

    homo = scf.get('e')[int(sum(mo_occ))-1]
    assert homo, -0.3912742463712855

    assert np.trace(scf.get('s')), scf.mol.natm

    '''
    the grid and associated weights are available from the cache via get('g') and get('w') respectively
    for full range of quantities available from .get() see the code in scf.rks.pt
    '''

#*******************************************************************************************************************************
#                                                            OUTPUT                                                            *
#*******************************************************************************************************************************
# ******************
# *   scf output   *
# ******************
# method                  RKS
# charge                  0
# spin                    0
# units                   bohr
# open shell              False      :  multiplicity  1
#
# basis is                sto-3g
# analytic integration    aello cython - McMurchie-Davidson scheme
#
# diis                    True  : buffer size is  6
#
# numerical integration   Mura-Knowles, Lebedev
#                         radial prune is Aldrich-Treutler
#                         Becke partitioning scheme is Stratmann
#                         radial adjustment is Treutler
#                         integration order  period 1 (10,11)   period 2 (15,15)
# mesh                    fine
#
# functional              LDA,VWN_RPA
# scf control             maximum cycles are  50    :  convergence tolerance  1e-07
#
# basis analysis
#--------------
# shell   l    pGTO    cGTO
# -------------------------
# O    0
#   0     0      3       1
#   1     0      3       1
#   2     1      3       1
# H    1
#   3     0      3       1
# H    2
#   4     0      3       1
#
# -------------------------
# number of shells            5
# number of primative GTO    21
# number of contracted GTO    7
#
#  cycle     1 electron         coulomb         exchange          electrons
#                                                                                    ΔE         diis norm
# -------------------------------------------------------------------------------------------------------------
#     1     -114.14210300     45.97016067     -9.91672337         10.0000
#                                                                                7.8089e+01    8.3888e-01
#     2     -124.91365638     45.30892828     -8.56834673         10.0000
#                                                                                1.0084e+01    9.3412e-01
#     3     -114.59104374     45.56240479     -9.64051915         10.0000
#                                                                                9.5039e+00    8.4171e-01
#     4     -124.80114732     45.50071447     -8.58377232         10.0000
#                                                                                9.2150e+00    9.1282e-01
#     5     -114.64920208     45.53382264     -9.61705049         10.0000
#                                                                                9.1518e+00    8.3094e-01
#     6     -122.33184746     44.32502262     -8.58591758         10.0000
#                                                                                7.8603e+00    9.0901e-01
#     7     -120.07475225     47.36508663     -9.21469855         10.0000
#                                                                                4.6684e+00    4.2751e-01
#     8     -120.23031888     46.18339332     -8.95425645         10.0000
#                                                                                1.0768e+00    3.6892e-02
#     9     -120.25951427     46.28479033     -8.96974954         10.0000
#                                                                                5.6709e-02    4.7781e-03
#    10     -120.25322758     46.29740978     -8.97271669         10.0000
#                                                                                1.5939e-02    1.2996e-03
#    11     -120.25327055     46.29397002     -8.97207678         10.0000
#                                                                                2.8428e-03    9.6453e-06
#    12     -120.25327381     46.29399549     -8.97208116         10.0000
#                                                                                1.7830e-05    6.7834e-07
#    13     -120.25327382     46.29399729     -8.97208149         10.0000
#                                                                                1.4666e-06    4.9638e-10
#    14     -120.25327382     46.29399729     -8.97208149         10.0000
#                                                                                1.0696e-09    1.8019e-13
#    15     -120.25327382     46.29399729     -8.97208149         10.0000
#                                                                                3.1264e-13    6.7081e-14
#
# nuclear repulsion      8.0023670618
# total electronic     -82.9313580134
#
# final total energy   -74.9289909516
#
# Geometry
# -----------
#  O       0.0000    -0.1432     0.0000
#  H       1.6380     1.1365     0.0000
#  H      -1.6380     1.1365     0.0000
#
# dipole momemts (Debye)
# ----------------------
#  x= -0.0000 y=  1.5029 z= -0.0000         resultant   1.5029
#
# Lowdin populations
# --------------------
#   [8.2021 0.8990 0.8990]
#
# charge
#   [-0.2021  0.1010  0.1010]        net = -0.00
#
# Mayer bond orders
# ------------------
#        H1        H2
# O0 | 0.9745    0.9745
# H1 |           0.0066
#
# Valency
# O0  = 1.95:   H1  = 0.98:   H2  = 0.98:
# True
