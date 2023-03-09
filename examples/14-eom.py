from __future__ import division
import numpy as np

from mol.mol import molecule
from scf.rhf import RHF
from phf.eom import EOM
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

    '''
    establish instance of solver class and instance of an EOM class
    '''

    solve = solver(roots=-1, vectors=True)
    eom = EOM(scf)

    '''
    EOM class has basic solvers for EE, IP and EA. The 'execute' method takes a first mandatory argument
    which can be one of 'ee'|'ip'|'ea'. The second mandatory arguement is a solver instance. Optional arguments
    are 'roots' and 'solution' with defaults 3 and 'iterative'. 'direct' will perform a dense matrix solution
    using a numpy or scipy eigensolver and 'iterative' will use a single targetted root Davidson eigensolver.
    The guess for the EE method is to use a CIS solution and for IP or EA to use a guess targetting Koopman
    states. The 'direct' solvers return many unphysical states due to the extra symmetry of the spin treatment.
    The results cache has keys :- 'values' (eigenvalues for eom.roots), 'vectors' - (eigenvectors for reduced
    space solution), 'quasi-particle weights' - (single space quasi-particle strength), 'norms' - (norms of
    singles and doubles spaces), 'excitations' - ([label, levels, values] for dominant excitations as controlled
    by set_options method with defaults 'all excitations > 0.1'. The keys 'norms' and 'excitations' are only
    available after the 'analyse()' method has been invoked. For EE the results cache has the key 'r zero' which
    returns the value of r0 which should be zero for states of different symmetry. The amplitudes can be obtained
    from the cached 'vectors' by using the .expand method as eg eom.expand(eom.cache['vectors'][:, i]). This will return
    r1 and r2 as shaped arrays. Note that the left-hand eigenvectors are available by using th key 'left=True' for
    IP and EA and for EE from the lambda section of the coupled-cluster class.
    '''

    def print_energies(eom):
        #format energies and norms and print

        print('\nEnergies, degeneracies and norms')
        print(' root     energy (au)       (eV)    multiplicity      quasi-particle weight\n','-'*74)
        pointer = 0
        for i, root in enumerate(eom.eom_degeneracies()):

            qpw = eom.cache['quasi-weights'][pointer]
            print(' {:2}      {:>10.6f}     {:>10.6f}       {:2}              {:>10.4f} '.
                format(i, root[0], root[0]*CONSTANTS('hartree->eV'), root[1], qpw))
            pointer += root[1]

    def print_excitations(eom):
        #format excitation details and print

        print('\nPrincipal excitations for ', end='')
        if eom.options['LEADING'] is None:     print('matrix dominant excitation > ', str(eom.options['THRESHOLD']))
        if not eom.options['LEADING'] is None: print('matrix dominant excitation - leading ',
                                                          str(eom.options['LEADING']), ' > ', str(eom.options['THRESHOLD']))
        print(' root     energy (au)     norm       block     excitation    ', '\n', '-'*60)
        pointer = 0
        for i, root in enumerate(eom.eom_degeneracies()):
            print(' {:2}      {:>10.6f}      {:>6.4f} '.format(i, root[0], eom.cache['norms'][pointer][0]))

            if np.isclose(eom.cache['norms'][pointer][0], 0): continue
            for excitation in eom.cache['excitations'][pointer]:

                print('                                     {:^6}'.format(excitation[0]), end='')
                for j in range(len(excitation[1])):
                    if j == 0:
                        print('   {:>6.4f} '.
                              format(excitation[1][j]), excitation[2][j], excitation[3])
                    else:
                        print('                                              {:>6.4f} '.
                              format(excitation[1][j]), excitation[2][j], excitation[3])

            pointer += root[1]

    #***********************
    #* Electron Excitation *
    #***********************

    eom.execute('ee', solve, roots=18, solution='iterative')
    eom.analyse()

    print('\n************************\n* EOM-CCSD-EE analysis *\n************************')
    print('\nRoots requested {:2}'.format(eom.roots))

    print_energies(eom)
    print_excitations(eom)

    #************************
    #* Ionisation Potential *
    #************************

    eom.execute('ip', solve, left=True, roots=eom.nocc, solution='iterative')
    eom.analyse()

    print('\n************************\n* EOM-CCSD-IP analysis *\n************************')
    print('\nRoots requested {:2}'.format(eom.roots))

    print_energies(eom)
    print_excitations(eom)

    #***********************
    #* Electron Attachment *
    #***********************

    eom.execute('ea', solve, roots=eom.nvir, solution='iterative')
    eom.analyse()

    print('\n************************\n* EOM-CCSD-EA analysis *\n************************')
    print('\nRoots requested {:2}'.format(eom.roots))

    print_energies(eom)
    print_excitations(eom)

'''
************************
* EOM-CCSD-EE analysis *
************************

Roots requested 18

Energies, degeneracies and norms
 root     energy (au)       (eV)    multiplicity      quasi-particle weight
 --------------------------------------------------------------------------
  0        0.275258       7.490149        3                  0.9435
  1        0.323244       8.795921        1                  0.9188
  2        0.361324       9.832139        3                  0.9384
  3        0.367942      10.012209        3                  0.9842
  4        0.394855      10.744542        1                  0.9211
  5        0.429243      11.680298        3                  0.9884
  6        0.496864      13.520353        1                  0.9258
  7        0.545324      14.839021        3                  0.9578

Principal excitations for matrix dominant excitation >  0.1
 root     energy (au)     norm       block     excitation
 ------------------------------------------------------------
  0        0.275258      0.9435
                                     1h-1p    0.6868  [5, 6] ⥣
  1        0.323244      0.9188
                                     1h-1p    0.6778  [5, 6] ⥣
  2        0.361324      0.9384
                                     1h-1p    0.6850  [5, 7] ⥮
  3        0.367942      0.9842
                                     1h-1p    0.6665  [4, 6] ⥮
                                              0.2114  [3, 7] ⥮
  4        0.394855      0.9211
                                     1h-1p    0.6786  [5, 7] ⥮
  5        0.429243      0.9884
                                     1h-1p    0.5615  [4, 7] ⥣
                                              0.4221  [3, 6] ⥣
  6        0.496864      0.9258
                                     1h-1p    0.6462  [4, 6] ⥮
                                              0.2009  [3, 7] ⥮
  7        0.545324      0.9578
                                     1h-1p    0.5517  [3, 6] ⥮
                                              0.4094  [4, 7] ⥮

************************
* EOM-CCSD-IP analysis *
************************

Roots requested 10

Energies, degeneracies and norms
 root     energy (au)       (eV)    multiplicity      quasi-particle weight
 --------------------------------------------------------------------------
  0        0.287506       7.823429        2                  0.9251
  1        0.391697      10.658619        2                  0.9428
  2        0.548698      14.930825        2                  0.9715
  3        1.220552      33.212909        2                  0.2622
  4       19.947897     542.809920        2                  0.8340

Principal excitations for matrix dominant excitation >  0.1
 root     energy (au)     norm       block     excitation
 ------------------------------------------------------------
  0        0.287506      0.9251
                                       1h     0.9618  [5]
  1        0.391697      0.9428
                                       1h     0.9697  [4]
  2        0.548698      0.9715
                                       1h     0.9856  [3]
  3        1.220552      0.2622
                                       1h     0.5120  [2]
  4       19.947897      0.8340
                                       1h     0.9132  [1]

************************
* EOM-CCSD-EA analysis *
************************

Roots requested  4

Energies, degeneracies and norms
 root     energy (au)       (eV)    multiplicity      quasi-particle weight
 --------------------------------------------------------------------------
  0        0.480962      13.087629        2                  0.9703
  1        0.578310      15.736628        2                  0.9505

Principal excitations for matrix dominant excitation >  0.1
 root     energy (au)     norm       block     excitation
 ------------------------------------------------------------
  0        0.480962      0.9703
                                       1p     0.9850  [6]
  1        0.578310      0.9505
                                       1p     0.9749  [7]
 '''
