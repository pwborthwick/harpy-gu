from __future__ import division
import numpy as np

from mol.mol import molecule, van_der_waals_radii
from scf.uks import UKS
if __name__ == '__main__':

    geo = ([['O', ( 0.000000000000, -0.143225816552, 0.000000000000)],
            ['H', ( 1.638036840407,  1.136548822547, 0.000000000000)],
            ['H', (-1.638036840407,  1.136548822547, 0.000000000000)]])

    # Computation options
    mol = molecule( geo,
                    spin=1,
                    units='bohr',
                    charge=1,
                    gto='sto-3g',
                    silent=False)

    scf = UKS(mol,
              xc='LDA,VWN_RPA',
              mesh='fine',
              cycles=50,
              tol=1e-7,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    scf.analyse('dipole,geometry,bonds,charge')

    #********************
    #* check with PySCF *
    #********************
    import numpy as np
    print(np.isclose(scf_energy, -74.57402544203664))

    #using cached values - 'x' is exchange-correlation potential
    core_hamiltonian = scf.get('t') + scf.get('v')
    eri = scf.get('i') ; d = scf.get('d')
    f = core_hamiltonian + np.einsum('rs,pqrs->pq', d[0]+d[1], eri, optimize=True) + scf.get('x')

    assert np.allclose(f, scf.get('f'))

    mo_coeff = scf.get('c') ; mo_occ = scf.get('o')
    density =  np.einsum('xpr,xr,xqr->xpq', mo_coeff, mo_occ, mo_coeff, optimize=True)

    assert np.allclose(density, d)

    homo_alpha, homo_beta = scf.get('e')[0][int(sum(mo_occ[0]))-1], scf.get('e')[1][int(sum(mo_occ[1]))-1]
    assert homo_alpha, -0.741562755824065
    assert homo_beta, -0.706976889742772

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
# method                  UKS
# charge                  1
# spin                    1
# units                   bohr
# open shell              True      :  multiplicity  2
#
# basis is                sto-3g
# analytic integration    aello cython - McMurchie-Davidson scheme
#
# diis                    True  : buffer size is  6
# scf control             maximum cycles are  50    :  convergence tolerance  1e-07
#
#  cycle     1 electron         coulomb         exchange          electrons           total
#                                                                     S²      multiplicity            ΔE         diis norm
# ----------------------------------------------------------------------------------------------------------------------------
#     1    -118.49493297      46.13689343     -9.11151923    (  5.0000,  4.0000)  -81.46955877
#     2    -109.30003255      35.52781495     -7.94662701    (  5.0000,  4.0000)  -81.71884460
#                                                                    0.751        2.001           -81.718845      1.093251
#     3    -117.73009689      44.48781602     -8.88036159    (  5.0000,  4.0000)  -82.12264246
#                                                                    0.750        2.000            -0.403798      1.437050
#     4    -110.25994883      36.27118778     -8.00365922    (  5.0000,  4.0000)  -81.99242027
#                                                                    0.751        2.001             0.130222      0.895889
#     5    -117.49887490      44.12675679     -8.83281195    (  5.0000,  4.0000)  -82.20493006
#                                                                    0.751        2.001            -0.212510      1.240307
#     6    -110.53613761      36.49658299     -8.02192392    (  5.0000,  4.0000)  -82.06147855
#                                                                    0.752        2.002             0.143452      0.839682
#     7    -114.57392726      40.38943068     -8.39178802    (  5.0000,  4.0000)  -82.57628459
#                                                                    0.753        2.003            -0.514806      1.176069
#     8    -114.62636627      40.44804612     -8.39807198    (  5.0000,  4.0000)  -82.57639212
#                                                                    0.753        2.003            -0.000108      0.017457
#     9    -114.62966000      40.45173609     -8.39846856    (  5.0000,  4.0000)  -82.57639247
#                                                                    0.753        2.003            -0.000000      0.000983
#    10    -114.62952649      40.45158697     -8.39845295    (  5.0000,  4.0000)  -82.57639247
#                                                                    0.753        2.003            -0.000000      0.000044
#    11    -114.62951434      40.45157344     -8.39845157    (  5.0000,  4.0000)  -82.57639247
#                                                                    0.753        2.003            -0.000000      0.000003
#
# nuclear repulsion      8.0023670618
# total electronic     -82.5763924665
#
# final total energy   -74.5740254047
#
# dipole momemts (Debye)
# ----------------------
#  α-spin           x= -0.0000 y=  1.6736 z=  0.0000         resultant   1.6736
#  β-spin           x= -0.0000 y=  1.0996 z= -0.0000         resultant   1.0996
#  α+β-spin         x= -0.0000 y=  2.7732 z= -0.0000         resultant   2.7732
#
# Geometry
# -----------
#  O       0.0000    -0.1432     0.0000
#  H       1.6380     1.1365     0.0000
#  H      -1.6380     1.1365     0.0000
#
# Mayer bond orders
# ------------------
#        H1        H2
# O0 | 0.8183    0.8183
# H1 |           0.0085
#
# Valency
# O0  = 1.64:   H1  = 0.83:   H2  = 0.83:
#
# Lowdin populations
# --------------------
# α spin      [4.3789 0.3106 0.3106]
# β spin      [3.3108 0.3446 0.3446]
#
# charge
# α+β spin    [0.3104 0.3448 0.3448]        net = 1.00
#
# True
