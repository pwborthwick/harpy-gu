from __future__ import division
import numpy as np

from mol.mol import molecule
from scf.rhf import RHF
from phf.adc import ADC
from phf.eig import solver

from mol.mol import CONSTANTS

if __name__ == '__main__':

    # Computation options
    mol = molecule([['O', ( 0.000000000000, -0.143225816552, 0.000000000000)],
                    ['H', ( 1.638036840407,  1.136548822547 ,0.000000000000)],
                    ['H', (-1.638036840407,  1.136548822547 ,0.000000000000)]],
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    def print_energies(adc):
        #format energies and norms and print

        print('\nEnergies, degeneracies and norms')
        print(' root     energy (au)       (eV)    multiplicity           norms\n','-'*76)
        pointer = 0
        for i, root in enumerate(adc.adc_degeneracies()):
            qp_weight = '*' if np.isclose(adc.cache['norms'][pointer][1], 0,0) else ''
            norms = adc.cache['norms'][pointer]

            print(' {:2}      {:>10.6f}     {:>10.6f}       {:2}    ({:5}) {:>6.4f}  ({:5}) {:>6.4f}  {:1}'.
                format(i, root[0], root[0]*CONSTANTS('hartree->eV'), root[1],
                       norms[0][0], norms[1], norms[0][1], norms[2], qp_weight))
            pointer += root[1]
        print('\'*\' - quasi-particle weight is zero')

    def print_excitations(adc):
        #format excitation details and print

        print('\nPrincipal excitations for ', end='')
        if adc.options['LEADING'] is None:     print('matrix dominant excitation > ', str(adc.options['THRESHOLD']))
        if not adc.options['LEADING'] is None: print('matrix dominant excitation - leading ',
                                                          str(adc.options['LEADING']), ' > ', str(adc.options['THRESHOLD']))
        print(' root     energy (au)     block     excitation    ', '\n', '-'*50)
        pointer = 0
        for i, root in enumerate(adc.adc_degeneracies()):
            print(' {:2}      {:>10.6f} '.format(i, root[0]))

            for excitation in adc.cache['excitations'][pointer]:
                print('                          {:6}'.format(excitation[0]), end='')
                for j in range(len(excitation[1])):
                    if j == 0:
                        print(' {:>6.4f} '.
                              format(excitation[1][j]), excitation[2][j], excitation[3])
                    else:
                        print('                                 {:>6.4f} '.
                              format(excitation[1][j]), excitation[2][j], excitation[3])

            pointer += root[1]

    def print_spectral(adc):
        #format spectral analysis and print

        print('\nSpectral analysis for dominant partial sum')
        print(' root     energy (au)     factor   dominant contribution    ', '\n', '-'*55)
        pointer = 0
        for i, root in enumerate(adc.adc_degeneracies()):
            print(' {:2}      {:>10.6f} '.format(i, root[0]), end='')
            excitation = adc.cache['spectral properties'][pointer]
            orbital = '' if np.isclose(excitation[2], 0.0) else excitation[1]
            print('      {:>6.4f}        {:<8.6f}  {:2}'.
                 format(excitation[0], excitation[2], orbital))
            pointer += root[1]


    '''
    establish instance of solver class and instance of an ADC class
    '''
    solve = solver(roots=-1, vectors=True)
    adc = ADC(scf)

    '''
    ADC class has a set_options method which allows control of reporting of dominant excitations. The
    'threshold' value will print all excitations with a value greater than that value. The 'leading' key
    will print the first n values. The keys can be combined so threshold=0.1, leading=5 will print the
    first 5 values if they are greater in value than 0.1.
    The 'execute' method takes a first mandatory argument which can be one of 'ee'|'ip'|'ea'. The second
    mandatory arguement is a solver instance. Optional arguments are 'roots' and 'solution' with defaults
    3 and 'direct'. 'direct' will perform a dense matrix solution using a numpy or scipy eigensolver.
    'iterative' will perform a Davidson solution. The method 'analyse' takes no arguments and performs
    an analysis of the solution. For all methods this is a listing of energies in atomic and electron volt
    units, the degeneracy of the root and the norms of the small and large spaces. For all methods the
    dominant excitation value and orbitals involved are printed along with whether the excitation involves
    same or different spin orbitals. For the 'ee' method transition dipole moments (electric length) are
    printed together with the associated oscillator strength. For'ip' and 'ea' methods the spectral factors
    are printed and the dominant spectral contribution. All computations are carried out at order 2 of
    Algebraic Diagramatic Construction.

    '''
    #***********************
    #* Electron Excitation *
    #***********************

    adc.set_options(threshold=0.03)
    adc.execute('ee', solve, roots=18, solution='iterative')
    adc.analyse()

    print('\n**********************\n* ADC(2)-EE analysis *\n**********************')
    print('\nRoots requested {:2}'.format(adc.roots))

    print_energies(adc)
    print_excitations(adc)

    print('\ntransition properties for singlets')
    print(' root     energy   dipole    x         y         z         oscillator   ', '\n', '-'*70)
    pointer = 0
    for i, root in enumerate(adc.adc_degeneracies()):
        if root[1] == 1:
            dipole, oscillator = adc.cache['transition properties'][pointer]
            print('  {:2}    {:>9.6f}        {:>8.4f}  {:>8.4f}  {:>8.4f}     {:>10.6f}'.
                  format(pointer, adc.cache['values'][pointer], dipole[0], dipole[1], dipole[2], oscillator))
        pointer += root[1]

    #************************
    #* Ionisation Potential *
    #************************

    adc.set_options(threshold=0.1)
    adc.execute('ip', solve, roots=8, solution='iterative')
    adc.analyse()

    print('\n**********************\n* ADC(2)-IP analysis *\n**********************')
    print('\nRoots requested {:2}'.format(adc.roots))

    print_energies(adc)
    print_excitations(adc)
    print_spectral(adc)

    #***********************
    #* Electron Attachment *
    #***********************

    adc.set_options(threshold=0.1)
    adc.execute('ea', solve, roots=6, solution='iterative')
    adc.analyse()

    print('\n**********************\n* ADC(2)-EA analysis *\n**********************')
    print('\nRoots requested {:2}'.format(adc.roots))

    print_energies(adc)
    print_excitations(adc)
    print_spectral(adc)

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

    **********************
    * ADC(2)-EE analysis *
    **********************

    Roots requested 18

    Energies, degeneracies and norms
     root     energy (au)       (eV)    multiplicity           norms
     ----------------------------------------------------------------------------
      0        0.285509       7.769103        3    (1h-1p) 0.9847  (2h-2p) 0.0153
      1        0.343916       9.358436        1    (1h-1p) 0.9784  (2h-2p) 0.0216
      2        0.363627       9.894796        3    (1h-1p) 0.9949  (2h-2p) 0.0051
      3        0.372292      10.130570        3    (1h-1p) 0.9884  (2h-2p) 0.0116
      4        0.414647      11.283113        1    (1h-1p) 0.9836  (2h-2p) 0.0164
      5        0.421080      11.458183        3    (1h-1p) 0.9984  (2h-2p) 0.0016
      6        0.504183      13.719528        1    (1h-1p) 0.9772  (2h-2p) 0.0228
      7        0.540957      14.720197        3    (1h-1p) 0.9917  (2h-2p) 0.0083
    '*' - quasi-particle weight is zero

    Principal excitations for matrix dominant excitation >  0.03
     root     energy (au)     block     excitation
     --------------------------------------------------
      0        0.285509
                              1h-1p  0.7017  [5, 6] ⥣
      1        0.343916
                              1h-1p  0.6994  [5, 6] ⥣
                              2h-2p  0.0333  [5, 3, 6, 7] ⥮
                                     0.0333  [3, 5, 7, 6] ⥮
      2        0.363627
                              1h-1p  0.6683  [4, 6] ⥮
                                     0.2170  [3, 7] ⥮
                                     0.0613  [2, 6] ⥮
      3        0.372292
                              1h-1p  0.7030  [5, 7] ⥮
      4        0.414647
                              1h-1p  0.7013  [5, 7] ⥮
      5        0.421080
                              1h-1p  0.5538  [4, 7] ⥣
                                     0.4377  [3, 6] ⥣
                                     0.0302  [2, 7] ⥣
      6        0.504183
                              1h-1p  0.6638  [4, 6] ⥮
                                     0.2094  [3, 7] ⥮
                                     0.0642  [2, 6] ⥮
                              2h-2p  0.0422  [4, 4, 6, 6] ⥮
      7        0.540957
                              1h-1p  0.5466  [3, 6] ⥮
                                     0.4353  [4, 7] ⥮
                                     0.0870  [2, 7] ⥮

    transition properties for singlets
     root     energy   dipole    x         y         z         oscillator
     ----------------------------------------------------------------------
       3     0.343916          0.0000   -0.0000   -0.0961       0.002119
      10     0.414647         -0.0000    0.0000    0.0000       0.000000
      14     0.504183         -0.0000    0.4328   -0.0000       0.062968

    **********************
    * ADC(2)-IP analysis *
    **********************

    Roots requested  8

    Energies, degeneracies and norms
     root     energy (au)       (eV)    multiplicity           norms
     ----------------------------------------------------------------------------
      0        0.272050       7.402859        2    (1h   ) 0.9125  (2h-1p) 0.0875
      1        0.377743      10.278915        2    (1h   ) 0.9415  (2h-1p) 0.0585
      2        0.536346      14.594708        2    (1h   ) 0.9719  (2h-1p) 0.0281
      3        1.103047      30.015442        2    (1h   ) 0.8042  (2h-1p) 0.1958
    '*' - quasi-particle weight is zero

    Principal excitations for matrix dominant excitation >  0.1
     root     energy (au)     block     excitation
     --------------------------------------------------
      0        0.272050
                              1h     0.9552  [5]
                              2h-1p  0.1359  [5, 3, 7] ⥣
                                     0.1359  [3, 5, 7] ⥣
                                     0.1322  [5, 4, 6] ⥣
                                     0.1322  [4, 5, 6] ⥣
                              2h-1p  0.1541  [3, 5, 7] ⥮
                                     0.1369  [4, 5, 6] ⥮
      1        0.377743
                              1h     0.9691  [4]
                              2h-1p  0.1272  [4, 3, 7] ⥣
                                     0.1272  [3, 4, 7] ⥣
                              2h-1p  0.1338  [4, 4, 6] ⥮
                                     0.1198  [3, 4, 7] ⥮
      2        0.536346
                              1h     0.9858  [3]
      3        1.103047
                              1h     0.8928  [2]
                              2h-1p  0.3115  [5, 5, 6] ⥮
                                     0.1408  [4, 3, 7] ⥮
                                     0.1274  [4, 4, 6] ⥮
                                     0.1218  [3, 4, 7] ⥮
                                     0.1117  [3, 2, 7] ⥮
                                     0.1054  [4, 2, 6] ⥮

    Spectral analysis for dominant partial sum
     root     energy (au)     factor   dominant contribution
     -------------------------------------------------------
      0        0.272050       0.9120        0.911954   4
      1        0.377743       0.9343        0.931787   3
      2        0.536346       0.9615        0.961477   2
      3        1.103047       0.8016        0.794836   1

    **********************
    * ADC(2)-EA analysis *
    **********************

    Roots requested  6

    Energies, degeneracies and norms
     root     energy (au)       (eV)    multiplicity           norms
     ----------------------------------------------------------------------------
      0        0.473847      12.894042        2    (1p   ) 0.9772  (1h-2p) 0.0228
      1        0.575860      15.669938        2    (1p   ) 0.9647  (1h-2p) 0.0353
      2        1.342824      36.540107        2    (1p   ) 0.0000  (1h-2p) 1.0000  *
    '*' - quasi-particle weight is zero

    Principal excitations for matrix dominant excitation >  0.1
     root     energy (au)     block     excitation
     --------------------------------------------------
      0        0.473847
                              1p     0.9885  [6]
      1        0.575860
                              1p     0.9822  [7]
                              1h-2p  0.1179  [4, 7, 6] ⥣
                                     0.1179  [4, 6, 7] ⥣
      2        1.342824
                              1h-2p  1.0000  [5, 6, 6] ⥮

    Spectral analysis for dominant partial sum
     root     energy (au)     factor   dominant contribution
     -------------------------------------------------------
      0        0.473847       0.9667        0.966638   6
      1        0.575860       0.9540        0.953943   7
      2        1.342824       0.0000        0.000000
'''
