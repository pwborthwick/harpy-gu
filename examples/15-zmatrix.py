from __future__ import division
import numpy as np

from mol.mol import molecule
from scf.rhf import RHF

'''
Our usual way of entering coordinates is with a list as first parameter to molecule instatiator
'''
if __name__ == '__main__':

    geo = ([['O', ( 0.000000000000, -0.143225816552, 0.000000000000)],
            ['H', ( 1.638036840407,  1.136548822547 ,0.000000000000)],
            ['H', (-1.638036840407,  1.136548822547 ,0.000000000000)]])

    # Computation options
    mol = molecule( geo,
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy_cartesian = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    '''
    Also supported is a z-matrix input where the first parameter to molecule is a string.
    It is advised to use the format as below. First the atom definitions using either the
    explicit values or symbolic substitutions as below. The symbol substitutions are lines
    of the form symbol = value. Value can be an 'eval'uatable expression. values are in the
    units specified by the 'units' keyword and angles are always in degrees. Dummy atoms can
    be used denoted by an atomic symbol 'X' - these will be removed after geometry has been
    constructed.
    The string can be entered as a single line using '\n' (and '\t') control characters if
    wished as in the H2 example.
    '''
    geo =(
'''
O
H   1   r
H   1   r   2   theta

r     = 2.0787
theta = 52.0 * 2.0
''')
    mol = molecule( geo,
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy_zmatrix = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    assert np.isclose(scf_energy_cartesian, scf_energy_zmatrix)

    mol = molecule( '''H\nH   1   0.74''',
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy_zmatrix = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    '''
    Finally an example of methane using
    C-H = 2.05au, bond angles=109.47, dihedrals=120
    '''
    geo =(
'''
C
H   1   r
H   1   r   2   theta
H   1   r   2   theta   3  120
H   1   r   2   theta   3  240

r     = 2.05
theta = 109.47
''')
    mol = molecule( geo,
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy_zmatrix = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

'''
    ******************
    *   scf output   *
    ******************
    method                  RHF
    charge                  0
    spin                    0
    units                   bohr
    open shell              False      :  multiplicity  1

    basis is                sto-3g
    analytic integration    aello cython - McMurchie-Davidson scheme

    diis                    True  : buffer size is  6
    scf control             maximum cycles are  50    :  convergence tolerance  1e-10

    basis analysis
    --------------
    shell   l    pGTO    cGTO
    -------------------------
    O    0
      0     0      3       1
      1     0      3       1
      2     1      3       1
    H    1
      3     0      3       1
    H    2
      4     0      3       1

    -------------------------
    number of shells            5
    number of primative GTO    21
    number of contracted GTO    7

     cycle        E SCF              ΔE            |diis|               homo            lumo
    ------------------------------------------------------------------------------------------------
        1     -78.28658323       7.8287e+01      8.4510e-01          0.30910705      0.55914745
        2     -84.04831633       5.7617e+00      3.5580e-01         -0.53724397      0.40720333
        3     -82.71696597       1.3314e+00      9.2429e-02         -0.34937966      0.49506621
        4     -82.98714079       2.7017e-01      1.9368e-02         -0.39652094      0.47684760
        5     -82.93813321       4.9008e-02      8.1314e-03         -0.38696302      0.47890075
        6     -82.94398045       5.8472e-03      3.4757e-03         -0.38752441      0.47760521
        7     -82.94444058       4.6013e-04      2.5198e-05         -0.38758528      0.47762524
        8     -82.94444697       6.3936e-06      4.7658e-06         -0.38758674      0.47761876
        9     -82.94444702       4.1294e-08      3.2206e-08         -0.38758674      0.47761872
       10     -82.94444702       1.4087e-10      9.4213e-11         -0.38758674      0.47761872
       11     -82.94444702       1.4211e-14      6.9327e-14         -0.38758674      0.47761872
       12     -82.94444702       4.2633e-14      6.8556e-14         -0.38758674      0.47761872

    nuclear repulsion      8.0023670618
    total electronic     -82.9444470159

    final total energy   -74.9420799540

    ******************
    *   scf output   *
    ******************
    method                  RHF
    charge                  0
    spin                    0
    units                   bohr
    open shell              False      :  multiplicity  1

    basis is                sto-3g
    analytic integration    aello cython - McMurchie-Davidson scheme

    diis                    True  : buffer size is  6
    scf control             maximum cycles are  50    :  convergence tolerance  1e-10

    basis analysis
    --------------
    shell   l    pGTO    cGTO
    -------------------------
    O    0
      0     0      3       1
      1     0      3       1
      2     1      3       1
    H    1
      3     0      3       1
    H    2
      4     0      3       1

    -------------------------
    number of shells            5
    number of primative GTO    21
    number of contracted GTO    7

     cycle        E SCF              ΔE            |diis|               homo            lumo
    ------------------------------------------------------------------------------------------------
        1     -78.28657216       7.8287e+01      8.4510e-01          0.30910742      0.55914706
        2     -84.04831619       5.7617e+00      3.5580e-01         -0.53724416      0.40720236
        3     -82.71695824       1.3314e+00      9.2430e-02         -0.34937932      0.49506574
        4     -82.98713558       2.7018e-01      1.9368e-02         -0.39652104      0.47684697
        5     -82.93812738       4.9008e-02      8.1315e-03         -0.38696300      0.47890016
        6     -82.94397473       5.8474e-03      3.4757e-03         -0.38752441      0.47760460
        7     -82.94443487       4.6013e-04      2.5198e-05         -0.38758528      0.47762463
        8     -82.94444126       6.3938e-06      4.7659e-06         -0.38758674      0.47761815
        9     -82.94444130       4.1294e-08      3.2206e-08         -0.38758674      0.47761811
       10     -82.94444130       1.4077e-10      9.4207e-11         -0.38758674      0.47761811
       11     -82.94444130       0.0000e+00      3.8786e-14         -0.38758674      0.47761811
       12     -82.94444130       0.0000e+00      3.8299e-14         -0.38758674      0.47761811

    nuclear repulsion      8.0023616239
    total electronic     -82.9444413010

    final total energy   -74.9420796772

    ******************
    *   scf output   *
    ******************
    method                  RHF
    charge                  0
    spin                    0
    units                   angstrom
    open shell              False      :  multiplicity  1

    basis is                sto-3g
    analytic integration    aello cython - McMurchie-Davidson scheme

    diis                    True  : buffer size is  6
    scf control             maximum cycles are  50    :  convergence tolerance  1e-10

    basis analysis
    --------------
    shell   l    pGTO    cGTO
    -------------------------
    H    0
      0     0      3       1
    H    1
      1     0      3       1

    -------------------------
    number of shells            2
    number of primative GTO     6
    number of contracted GTO    2

     cycle        E SCF              ΔE            |diis|               homo            lumo
    ------------------------------------------------------------------------------------------------
        1      -1.83186365       1.8319e+00      4.1792e-16         -0.57855386      0.67114348
        2      -1.83186365       8.8818e-16      4.1792e-16         -0.57855386      0.67114348
        3      -1.83186365       1.5543e-15      1.0448e-16         -0.57855386      0.67114348

    nuclear repulsion      0.7151043391
    total electronic      -1.8318636466

    final total energy    -1.1167593075

    ******************
    *   scf output   *
    ******************
    method                  RHF
    charge                  0
    spin                    0
    units                   bohr
    open shell              False      :  multiplicity  1

    basis is                sto-3g
    analytic integration    aello cython - McMurchie-Davidson scheme

    diis                    True  : buffer size is  6
    scf control             maximum cycles are  50    :  convergence tolerance  1e-10

    basis analysis
    --------------
    shell   l    pGTO    cGTO
    -------------------------
    C    0
      0     0      3       1
      1     0      3       1
      2     1      3       1
    H    1
      3     0      3       1
    H    2
      4     0      3       1
    H    3
      5     0      3       1
    H    4
      6     0      3       1

    -------------------------
    number of shells            7
    number of primative GTO    27
    number of contracted GTO    9

     cycle        E SCF              ΔE            |diis|               homo            lumo
    ------------------------------------------------------------------------------------------------
        1     -47.64756274       4.7648e+01      1.5187e+00          0.11829762      1.09102900
        2     -54.26878760       6.6212e+00      5.2461e-01         -0.62678505      0.58617891
        3     -53.03642369       1.2324e+00      9.1176e-02         -0.49936301      0.73976735
        4     -53.25984944       2.2343e-01      1.5990e-02         -0.52329079      0.71368821
        5     -53.22060747       3.9242e-02      2.9131e-03         -0.51914494      0.71830139
        6     -53.22614258       5.5351e-03      4.9538e-04         -0.51973244      0.71761679
        7     -53.22648041       3.3783e-04      2.4983e-05         -0.51977690      0.71760788
        8     -53.22647934       1.0683e-06      5.9952e-08         -0.51977681      0.71760796
        9     -53.22647934       1.3903e-10      1.4515e-11         -0.51977681      0.71760796
       10     -53.22647934       4.4054e-13      1.7948e-13         -0.51977681      0.71760796
       11     -53.22647934       1.4211e-14      4.4123e-14         -0.51977681      0.71760796

    nuclear repulsion     13.4996266414
    total electronic     -53.2264793414

    final total energy   -39.7268527000
'''
