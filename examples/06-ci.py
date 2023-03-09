from mol.mol import molecule
from scf.rhf import RHF
import int.mo_spin as mos
from phf.eig import solver
from mol.mol import CONSTANTS

import phf.cit as cit

import numpy as np

if __name__ == '__main__':
    #we're using the geometry from the Crawford Projects
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

#*****************************
#* Configuration Interaction *
#*****************************

# get instance of Configuration Interaction class
    ci = cit.CI(scf)

    def info(ci):
        #print general information

        print('\nrequested level of CI is {:1} - implementing [{:10}]'.
             format(ci.statistics['requested level'], ci.statistics['method']))
        print('full FCI with {:2} electrons in {:2} spin-orbitals has [{:8}] determinants'.
             format(sum(ci.scf.mol.nele), 2*ci.scf.mol.norb, ci.statistics['FCI determinant space']))
        print('effective number of determinants used [{:8}]'.
             format(ci.statistics['effective space']))

# note the Hamiltonian includes the ground state electronic energy, if you just want the excitation levels subtract the SCF ground state
# energy times a unit matrix of same dimenstion as Hamiltonian

# start instance of solver class
    solve = solver(roots=16, vectors=True)

#CIS
# invoke the fci method of CI class using a spin_adapted determinant basis - this is always faster.
    ci.FCI('S', spin_adapt=True)
    info(ci)
    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('first excitation {:<10.8f}'.format(solve.values[0] - scf.reference[0]))

# here string '0011111 1001111' ,which we read from right to left as an excitation of the beta-spin electron in spin orbital 5
# 1 to the virtual beta-spin orbital of  7.
    n = 10
    print('\nDominant matrix elements\n-------------------------\n  magnitude                   det a                            det b')
    dominant_elements = ci.get_fci_dominant(n)
    for i in dominant_elements:
        print(' {:>10.6f}      '.format(i[0]), i[1], '    ',i[2])

# check against Crawford values
    n = 8
    verify =  np.allclose([i- scf.reference[0] for i in solve.values[:8]], 
                         [0.2872554996, 0.3444249963, 0.3564617587, 0.3659889948, 0.3945137992, 0.4160717386, 0.5056282877, 0.5142899971])
    assert verify

#CISD
    ci.FCI('D')
    info(ci)

    solve.roots = 1
    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('ground state {:7} electronic energy  {:>12.8f} Hartree            correction {:>12.8f}'.
              format(ci.statistics['method'], solve.values[0], solve.values[0] - scf.reference[0]))

#CISDT
    ci.FCI('T')
    info(ci)

    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('ground state {:7} electronic energy  {:>12.8f} Hartree            correction {:>12.8f}'.
            format(ci.statistics['method'], solve.values[0], solve.values[0] - scf.reference[0]))
#CISDT
    ci.FCI('Q')
    info(ci)

    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('ground state {:7} electronic energy  {:>12.8f} Hartree            correction {:>12.8f}'.
             format(ci.statistics['method'], solve.values[0], solve.values[0] - scf.reference[0]))

    sdtq = solve.values[0]
# and to show for this computation CISDTQ = FCI
#F
    ci.FCI('F')
    info(ci)

    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('ground state {:7} electronic energy  {:>12.8f} Hartree            correction {:>12.8f}'.
            format(ci.statistics['method'], solve.values[0], solve.values[0] - scf.reference[0]))

        assert sdtq, solve.values[0]

#*******
#* CIS *
#*******
# standard implementation of CIS, ci.CIS returns the Hamiltonian then envoke the solver with 'roots=-1' which
# means all roots. Can pass eigenvalues through ci.degeneracies which will give a list of tupes (energy, multiplicity)
    print('\nConfiguration Interaction Singles\n-----------------------------')

    solve.roots = -1
    ci.CIS()
    solve.direct(ci.hamiltonian)
    if solve.converged:
        degen_tuples = ci.degeneracies(solve.values)
        print('\ndegeneracy analysis of CIS eigenvalues\n     energy   degeneracy\n------------------------')
        for i in degen_tuples:
            print(' {:>12.8f}    {:2}  '.format(i[0], i[1]))

#**************************************************
#* spin-adapted Configuration Interaction Singles *
#**************************************************
# spin-adapted CIS which give siglets and triplets separately.
    print('\nspin-adapted CIS\n----------------')

# singlets
    solve.roots = 5
    ci.spin_adapted_CIS(type='singlet')
    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('first ', n, ' singlets are ', solve.values)

# triplets
    ci.spin_adapted_CIS(type='triplet')
    solve.direct(ci.hamiltonian)
    if solve.converged:
        print('first ', n, ' triplets are ', solve.values)

#******************************
#* Random Phase Approximation *
#******************************
# RPA solutions in various forms
    print('\nRandom Phase Approximation\n--------------------------')

#using black matrix structure of A and B sub-matrices block[[A, B], [-B, -A]]
    solve.roots = -1
    method = 'block'
    ci.RPA(type=method)
    solve.direct(ci.hamiltonian)
    if solve.converged:
        positive_eigenvalues = solve.values[solve.values > 0][:n] 
        print('first ', n, ' positive eigenvalues by  [', method, ']    method are (eV)', positive_eigenvalues * CONSTANTS('hartree->eV'))

#using linear matrix structure of A and B sub-matrices - dot(A+B, A-B)
    method = 'linear'
    ci.RPA(type=method)
    solve.direct(ci.hamiltonian)
    if solve.converged:
        positive_eigenvalues = solve.values[solve.values > 0][:n] 
        print('first ', n, ' positive eigenvalues by [', method, ']    method are (eV)', positive_eigenvalues * CONSTANTS('hartree->eV'))

#using the Hermiian (A-B)^(1/2).(A+B).(A-B)^(1/2)
    method = 'hermitian'
    ci.RPA(type=method)
    solve.direct(ci.hamiltonian)
    if solve.converged:
        positive_eigenvalues = solve.values[solve.values > 0][:n] 
        print('first ', n, ' positive eigenvalues by [', method, '] method are (eV)', positive_eigenvalues * CONSTANTS('hartree->eV'))

#using the Tamm-Dancoff Apptoximation
    method = 'TDA'
    ci.RPA(type=method)
    solve.direct(ci.hamiltonian)
    if solve.converged:
        positive_eigenvalues = solve.values[solve.values > 0][:n] 
        print('first ', n, ' positive eigenvalues by [', method, ']       method are (eV)', positive_eigenvalues * CONSTANTS('hartree->eV'))

#if you just want the excitation(A) and de-excitation(B) matrices use
    A, B = ci.RPA(type='AB')
    print('\nshape of A and B matrices is ', A.shape)

#we can get the degeneracy by evoking the degeneracy method which returns a list of (value, multiplicity) tuples
    positive_eigenvalues = solve.values[solve.values > 0]
    degen_tuples = ci.degeneracies(positive_eigenvalues)
    print('\ndegeneracy analysis of RPA-TDA eigenvalues\n     energy   degeneracy\n------------------------')
    for i in degen_tuples:
        print(' {:>12.8f}    {:2}  '.format(i[0], i[1]))

#***********
#* CIS-MP2 *
#***********
# CIS with MP2 corrected roots

#get CIS eigensolution - corrections written to list of lists [CIS, MP2 correction]
    ci.CIS_MP2()

    print('\nCIS-MP2 correction\n------------------')
    print('\n root      CIS          CIS-MP2        \u0394 \n---------------------------------------------')
    for root, i in enumerate(ci.correction):
        print('  {:>2d}   {:>10.6f}    {:>10.6f} ({:>10.6f} )'.
                    format(root+1, i[0], sum(i), i[1]))

#**********
#* CIS(D) *
#**********
#CIS(D) method
    ci.CIS_D()

    print('\nCIS(D) correction\n------------------')
    print('\n root      CIS          CIS(D)         \u0394 \n---------------------------------------------')
    for root, i in enumerate(ci.correction):
        print('  {:>2d}   {:>10.6f}    {:>10.6f} ({:>10.6f} )'.
                    format(root+1, i[0], sum(i), i[1]))

#************************
# Transition Properties *
#************************
# we can get the electric transition dipole and oscillator strengths from CIS or RPA('TDA')

    print('\nCIS Transition properties\n-------------------------')
    ci.CIS()

    solve.vectors = True
    solve.direct(ci.hamiltonian)

#transition properties will return only roots with a non-zero oscillator strength
    if solve.converged and ci.transition_method == 'cis':

        properties = ci.transition_properties(solve, roots=-1)
        print('\nroot      energy                dipole              oscillator       excitation\n', '-'*80)
        for i in properties:
            print(' {:2}    {:>10.6f}    {:>8.4f} {:>8.4f} {:>8.4f}     {:>8.4f}        {:10}'.
                                       format(i[0], i[1], i[2][0], i[2][1], i[2][2], i[3], i[4]))

#from FCI('S')
    ci.FCI('S')
    solve.direct(ci.hamiltonian)

    if solve.converged and ci.transition_method == 'fci':
        properties = ci.transition_properties(solve, roots=-1)
        print('\ntransition properties from FCI(\'S\')')
        print('\nroot      energy                dipole              oscillator       excitation\n', '-'*80)
        for i in properties:
            print(' {:2}    {:>10.6f}    {:>8.4f} {:>8.4f} {:>8.4f}     {:>8.4f}        {:10}'.
                                       format(i[0], i[1], i[2][0], i[2][1], i[2][2], i[3], i[4]))

#from spin-adapted singlets
    solve.roots = 5
    ci.spin_adapted_CIS(type='singlet')
    solve.direct(ci.hamiltonian)
    properties = ci.transition_properties(solve, roots=solve.roots)

    if solve.converged and ci.transition_method == 'cis':
        print('\ntransition properties from spin-adapted singlets')
        print('\nroot      energy                dipole              oscillator       excitation\n', '-'*80)
        for i in properties:
            print(' {:2}    {:>10.6f}    {:>8.4f} {:>8.4f} {:>8.4f}     {:>8.4f}        {:10}'.
                                       format(i[0], i[1], i[2][0], i[2][1], i[2][2], i[3], i[4]))

#*******************************
#* Davidson Iterative Solution *
#*******************************
# first test if CI method has a methods for iterative solution using ci.davidson, pass CI object to solver
    ci.CIS()
    if ci.davidson:
        solve.roots = 8
        solve.iterative(ci)
        
        if solve.converged:

            print('\nCIS Davidson\n------------------')
            print('\n root   CIS  energy\n-----------------------')
            for root, i in enumerate(solve.values):
                print('  {:>2d}    {:>10.6f}'.format(root+1, i))


    ci.spin_adapted_CIS(type='singlet')
    if ci.davidson:
        solve.iterative(ci)

        if solve.converged:

            print('\nspin-adapted singlets CIS Davidson\n------------------')
            print('\n root   CIS  energy\n-----------------------')
            for root, i in enumerate(solve.values):
                print('  {:>2d}    {:>10.6f}'.format(root+1, i))

    ci.spin_adapted_CIS(type='triplet')
    if ci.davidson:
        solve.iterative(ci)

        if solve.converged:

            print('\nspin-adapted triplets CIS Davidson\n------------------')
            print('\n root   CIS  energy\n-----------------------')
            for root, i in enumerate(solve.values):
                print('  {:>2d}    {:>10.6f}'.format(root+1, i))

#******************************
#using residue lists from FCI *
#******************************
#The FCI method has a 'use_residues' option which will generate the determinant basis using residues. It supports
#'S', 'D' and 'SD' CI types. If '*' is included in the string no Hamiltonian generation occurs and the determinant
#basis can be obtained from the 'determinant_list' propertty

#CIS
ci.FCI(spin_adapt=True, use_residues='S')

solve.roots, solve.vectors = 1, False
solve.direct(ci.hamiltonian)
if solve.converged:
    print('\nCIS using singles residues from FCI              {:>12.8f} '.format(solve.values[0] - scf.reference[0]))

#CID
ci.FCI(spin_adapt=True, use_residues='D')

solve.direct(ci.hamiltonian)
if solve.converged:
    print('CID using singles and doubles residues from FCI  {:>12.8f} '.format(solve.values[0] - scf.reference[0]))

#CISD
ci.FCI(spin_adapt=True, use_residues='SD')

solve.direct(ci.hamiltonian)
if solve.converged:
    print('CISD using singles and doubles residues from FCI {:>12.8f} '.format(solve.values[0] - scf.reference[0]))

#we can determine the multiplicity of spin-adapted root (only) using the 'is_singlet' method - needs vectors
print('\nMultiplicity of spin-adapted energies using -is_singlet- method')
ci.FCI(spin_adapt=True, use_residues='S')

solve.roots, solve.vectors = 10, True
solve.direct(ci.hamiltonian)
if solve.converged:
    for root, energy in enumerate(solve.values):
        if ci.is_singlet(solve.vectors[:, root]):
            print('root {:2} with value {:>8.6f} is a singlet'.format(root, solve.values[root] - scf.reference[0]))
        else:
            print('root {:2} with value {:>8.6f} is a triplet'.format(root, solve.values[root] - scf.reference[0]))

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
       10     -82.94444702       1.4070e-10      9.4176e-11         -0.38758674      0.47761872   
       11     -82.94444702       0.0000e+00      1.8538e-14         -0.38758674      0.47761872   
       12     -82.94444702       2.8422e-14      2.0764e-14         -0.38758674      0.47761872   

    nuclear repulsion      8.0023670618
    total electronic     -82.9444470159

    final total energy   -74.9420799540

    requested level of CI is S - implementing [S         ]
    full FCI with 10 electrons in 14 spin-orbitals has [1001    ] determinants
    effective number of determinants used [20      ]
    first excitation 0.28725552

    Dominant matrix elements
    -------------------------
      magnitude                   det a                            det b
      20.030856       | (α) 0011111  (β) 1011110>      | (α) 0011111  (β) 1011110>
      20.030856       | (α) 1011110  (β) 0011111>      | (α) 1011110  (β) 0011111>
      19.984700       | (α) 0111110  (β) 0011111>      | (α) 0111110  (β) 0011111>
      19.984700       | (α) 0011111  (β) 0111110>      | (α) 0011111  (β) 0111110>
       1.256657       | (α) 1011101  (β) 0011111>      | (α) 1011101  (β) 0011111>
       1.256657       | (α) 0011111  (β) 1011101>      | (α) 0011111  (β) 1011101>
       1.188332       | (α) 0111101  (β) 0011111>      | (α) 0111101  (β) 0011111>
       1.188332       | (α) 0011111  (β) 0111101>      | (α) 0011111  (β) 0111101>
       0.712216       | (α) 1011011  (β) 0011111>      | (α) 1011011  (β) 0011111>
       0.712216       | (α) 0011111  (β) 1011011>      | (α) 0011111  (β) 1011011>

    requested level of CI is D - implementing [SD        ]
    full FCI with 10 electrons in 14 spin-orbitals has [1001    ] determinants
    effective number of determinants used [311     ]
    ground state SD      electronic energy  -83.01359009 Hartree            correction  -0.06914307

    requested level of CI is T - implementing [SDT       ]
    full FCI with 10 electrons in 14 spin-orbitals has [1001    ] determinants
    effective number of determinants used [791     ]
    ground state SDT     electronic energy  -83.01372867 Hartree            correction  -0.06928165

    requested level of CI is Q - implementing [SDTQ      ]
    full FCI with 10 electrons in 14 spin-orbitals has [1001    ] determinants
    effective number of determinants used [1001    ]
    ground state SDTQ    electronic energy  -83.01534729 Hartree            correction  -0.07090027

    requested level of CI is F - implementing [SDTQ      ]
    full FCI with 10 electrons in 14 spin-orbitals has [1001    ] determinants
    effective number of determinants used [1001    ]
    ground state SDTQ    electronic energy  -83.01534729 Hartree            correction  -0.07090027

    Configuration Interaction Singles
    -----------------------------

    degeneracy analysis of CIS eigenvalues
         energy   degeneracy
    ------------------------
       0.28725552     3  
       0.34442501     3  
       0.35646178     1  
       0.36598901     3  
       0.39451381     3  
       0.41607175     1  
       0.50562830     1  
       0.51429000     3  
       0.55519189     1  
       0.56305577     3  
       0.65531846     1  
       0.91012170     1  
       1.10877096     3  
       1.20009613     3  
       1.30078519     1  
       1.32576205     1  
      19.95852619     3  
      20.01097921     1  
      20.01134187     3  
      20.05053172     1  

    spin-adapted CIS
    ----------------
    first  8  singlets are  [0.35646178 0.41607175 0.5056283  0.55519189 0.65531846]
    first  8  triplets are  [0.28725552 0.34442501 0.36598901 0.39451381 0.51429   ]

    Random Phase Approximation
    --------------------------
    first  8  positive eigenvalues by  [ block ]    method are (eV) [7.75970044 7.75970044 7.75970044 8.15643494 8.15643494 8.15643494
     9.5954604  9.5954604 ]
    first  8  positive eigenvalues by [ linear ]    method are (eV) [7.75970044 7.75970044 7.75970044 8.15643494 8.15643494 8.15643494
     9.5954604  9.5954604 ]
    first  8  positive eigenvalues by [ hermitian ] method are (eV) [7.75970044 7.75970044 7.75970044 8.15643494 8.15643494 8.15643494
     9.5954604  9.5954604 ]
    first  8  positive eigenvalues by [ TDA ]       method are (eV) [7.81662074 7.81662074 7.81662074 9.37228185 9.37228185 9.37228185
     9.69981897 9.95906814]

    shape of A and B matrices is  (40, 40)

    degeneracy analysis of RPA-TDA eigenvalues
         energy   degeneracy
    ------------------------
       0.28725552     3  
       0.34442501     3  
       0.35646178     1  
       0.36598901     3  
       0.39451381     3  
       0.41607175     1  
       0.50562830     1  
       0.51429000     3  
       0.55519189     1  
       0.56305577     3  
       0.65531846     1  
       0.91012170     1  
       1.10877096     3  
       1.20009613     3  
       1.30078519     1  
       1.32576205     1  
      19.95852619     3  
      20.01097921     1  
      20.01134187     3  
      20.05053172     1  

    CIS-MP2 correction
    ------------------

     root      CIS          CIS-MP2        Δ 
    ---------------------------------------------
       1     0.287256      0.241396 ( -0.045860 )
       2     0.287256      0.241396 ( -0.045860 )
       3     0.287256      0.241396 ( -0.045860 )
       4     0.344425      0.318999 ( -0.025426 )
       5     0.344425      0.318999 ( -0.025426 )
       6     0.344425      0.318999 ( -0.025426 )
       7     0.356462      0.299056 ( -0.057405 )
       8     0.365989      0.328162 ( -0.037827 )
       9     0.365989      0.328162 ( -0.037827 )
      10     0.365989      0.328162 ( -0.037827 )

    CIS(D) correction
    ------------------

     root      CIS          CIS(D)         Δ 
    ---------------------------------------------
       1     0.287256      0.236333 ( -0.050923 )
       2     0.287256      0.236333 ( -0.050923 )
       3     0.287256      0.236333 ( -0.050923 )
       4     0.344425      0.314683 ( -0.029742 )
       5     0.344425      0.314683 ( -0.029742 )
       6     0.344425      0.314683 ( -0.029742 )
       7     0.356462      0.294488 ( -0.061974 )
       8     0.365989      0.323216 ( -0.042773 )
       9     0.365989      0.323216 ( -0.042773 )
      10     0.365989      0.323216 ( -0.042773 )

    CIS Transition properties
    -------------------------

    root      energy                dipole              oscillator       excitation
     --------------------------------------------------------------------------------
      6      0.356462      0.0000   0.0000  -0.0993       0.0023        4 -> 5  (70%)
     14      0.505628     -0.0000   0.4389   0.0000       0.0649        3 -> 5  (66%)
     18      0.555192     -0.2044  -0.0000  -0.0000       0.0155        3 -> 6  (58%)
     22      0.655318     -1.6928   0.0000  -0.0000       1.2519        2 -> 5  (58%)
     23      0.910122     -0.0000  -1.1828  -0.0000       0.8488        2 -> 6  (65%)
     30      1.300785     -0.0000  -0.3264  -0.0000       0.0924        1 -> 5  (69%)
     31      1.325762     -0.0327  -0.0000   0.0000       0.0009        1 -> 6  (70%)
     35     20.010979     -0.0000   0.0656  -0.0000       0.0574        0 -> 5  (70%)
     39     20.050532      0.0788   0.0000   0.0000       0.0829        0 -> 6  (70%)

    transition properties from FCI('S')

    root      energy                dipole              oscillator       excitation
     --------------------------------------------------------------------------------
      6      0.356462      0.0000  -0.0000  -0.0993       0.0023        4 -> 5  (70%)
     14      0.505628     -0.0000  -0.4389   0.0000       0.0649        3 -> 5  (66%)
     18      0.555192     -0.2044   0.0000  -0.0000       0.0155        3 -> 6  (58%)
     22      0.655318      1.6928   0.0000   0.0000       1.2519        2 -> 5  (58%)
     23      0.910122      0.0000   1.1828  -0.0000       0.8488        2 -> 6  (65%)
     30      1.300785      0.0000  -0.3264   0.0000       0.0924        1 -> 5  (69%)
     31      1.325762      0.0327   0.0000  -0.0000       0.0009        1 -> 6  (70%)
     35     20.010979     -0.0000  -0.0656   0.0000       0.0574        0 -> 5  (70%)
     39     20.050532     -0.0788   0.0000  -0.0000       0.0829        0 -> 6  (70%)

    transition properties from spin-adapted singlets

    root      energy                dipole              oscillator       excitation
     --------------------------------------------------------------------------------
      0      0.356462      0.0000   0.0000  -0.0993       0.0023        4 -> 5  (99%)
      2      0.505628     -0.0000   0.4389   0.0000       0.0649        3 -> 5  (93%)
      3      0.555192     -0.2044  -0.0000  -0.0000       0.0155        3 -> 6  (83%)
      4      0.655318      1.6928  -0.0000   0.0000       1.2519        2 -> 5  (83%)

    CIS Davidson
    ------------------

     root   CIS  energy
    -----------------------
       1      0.287256
       2      0.287256
       3      0.287256
       4      0.344425
       5      0.344425
       6      0.356462
       7      0.365989
       8      0.365989

    spin-adapted singlets CIS Davidson
    ------------------

     root   CIS  energy
    -----------------------
       1      0.356462
       2      0.416072
       3      0.505628
       4      0.555192
       5      0.655318
       6      0.910122
       7      1.300785
       8      1.325762

    spin-adapted triplets CIS Davidson
    ------------------

     root   CIS  energy
    -----------------------
       1      0.287256
       2      0.344425
       3      0.365989
       4      0.394514
       5      0.514290
       6      0.563056
       7      1.108771
       8      1.200096

    CIS using singles residues from FCI                0.28725552 
    CID using singles and doubles residues from FCI   -0.06865825 
    CISD using singles and doubles residues from FCI  -0.06914307 

    Multiplicity of spin-adapted energies using -is_singlet- method
    root  0 with value 0.287256 is a triplet
    root  1 with value 0.344425 is a triplet
    root  2 with value 0.356462 is a singlet
    root  3 with value 0.365989 is a triplet
    root  4 with value 0.394514 is a triplet
    root  5 with value 0.416072 is a singlet
    root  6 with value 0.505628 is a singlet
    root  7 with value 0.514290 is a triplet
    root  8 with value 0.555192 is a singlet
    root  9 with value 0.563056 is a triplet

'''
