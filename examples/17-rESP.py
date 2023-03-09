from __future__ import division
import numpy as np

from mol.mol import molecule, van_der_waals_radii
from scf.rhf import RHF
from phf.rESP import RESP

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
                    silent=True)

    scf = RHF(mol,
              cycles=50,
              tol=1e-7,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    print('','*'*39, '\n * restrained Electrostatic Potentials * \n','*'*39)
    print('Example 1 - water\n')

    '''
    Initiate the RESP class by passing a RHF clas object to the class initiator, this sets the default options
    '''
    esp = RESP(scf)

    '''
    Our object is to generate a distribution of points at which to evaluate the electrostatic potential. This is achieved
    by defining a system of shells around each atom on which the points are located. This system can be a single spherical
    shell of a nested sequence of shells. All atoms have the same shell system. The number of shell around each atom is
    given by the parameter keyword 'count' and can be set by the set_parameters method as .set_parameters(count=n). The shells
    are positioned at a radius which is a multiple of the Van der Waals radius for that atoms species. By default the first
    shell is positioned at the Van der Waals radius, this can be changed with the 'scale' parameter keyword eg
    .set_parameters(scale=1.4) will position the first shell at 1.4 * the Van der Waals radius. If the shell 'count' is more
    than 1 the shells are separated by a distance given by keyword 'increment' which has a default of 0.2 (all measurements in
    Angstroms). So to set a nested series of four shells starting at 1.2 times the Van der Waals radius and separated by 0.2
    Angstroms we would set .set_parameters(count=4, scale=1.2, increment=0.2)
    '''
    esp.set_parameters(count=4, increment=0.2, scale=1.4 )

    '''
    There are two schemes implemented for distributing the points over the surfaces of the shells either Connolly or Fibonacci. These
    are set with the 'distribution' keyword with options 'connolly'|'fibonacci|saff and kuijlars'. The density of points is determined
    with the 'density' or 'points' keywords. 'density' sets the number of points in each square Anstrom of the shell surface while
    'points' sets the total number of points to distributed over the shell surface according to the 'distribution' algorithm.
    '''
    esp.set_parameters(distribution='connolly', density=1)

    '''
    The default Van der Waals radii can be modified with the 'radii' keyword as .set_parameters(radii= [['O', 1.40], ['H', 1.2]])
    '''
    esp.set_parameters(radii=[ ['O', 1.40], ['H', 1.2] ])

    '''
    Optionally it is possible to constrain the charges on some atoms. For example we can constrain a group of atoms to have equal charges.
    the keyword 'constain' is used as .set_parameters(constrain=[ ['=', [2, 3, 4] ] ]). This will force atoms 2,3 and 4 (base 1) to have
    the same charge. For water we might constrain the two hydrogens to have the same charge with
    .set_parameters('constrain = ['=', [2, 3]]]). It is also possible to constrain atoms so that the sum of their charges equals a
    predetermined value using .set_parameters(constrain=[ [-1.0, [5, 6]] ]). This will force the atoms 5 and 6 to have a combined charge
    of -1. This might be used for COO- groups in a zwitterion amino acid. If only one atom is specified the it will be constrained to the
    given charge eg [-0.8, [2]] will coerse atom [2] to have a charge of -0.8. It is not possible to coerse all atoms.
    '''
    esp.set_parameters(constrain=[ ['=', [2,3]] ])

    '''
    To run the calculation use the .execute() method
    '''
    esp.execute()

    '''
    After running the calculation the parameters are available as a dictionary in the 'parameters' property of the class. The results are
    available in a cache property. This has keys 'free' if no constraints are applied otherwise 'constrained' returns the computed charges as
    an array. The root mean square differences between the rESP computation and a classical q/r computation are returned in a dictionary with
    cache key 'root mean square' - this dictionary has keys 'free' or 'constrained'.
    '''
    type = 'free' if not 'constrain' in esp.parameters else 'constrained'
    print('          ', end='')
    for i in scf.mol.atom:
        print('  {:^12}'.format(i.symbol + str(i.id+1)), end='')
    print('     \u0394rms\n', '{:10}'.format(type), end='')
    for i in esp.cache[type]:
        print('{:^12.4f}  '.format(i), end='')
    print('{:>8.4e}'.format(esp.cache['root mean square'][type]))

    '''
    The restrained electrostatic potential approach employs a hyperbolic restraint directed towards the origin.
    This imposes a penalty on charges with high magnitudes. The intention is to reduce overfitting and so smooth the
    variation between charges, resulting in charges that are more conformationally independent. Restrained charges
    can be deployed using the 'restrain' keyword. Fitting parameters can be specifed for the matrix (a) and vector (b)
    with defaults being 0.0005 and 0.1 for the Lagrange multiplier system Aq=b. It is possible to exclude hydrogens
    from the restraint process with the keyword 'H'=False they are included by default. Optionally a maximum number
    of iterations and a convergence tolerance for the restraint iterative process can be given by keywords 'cycles'
    and 'tol' with defaults scf.cycles and scf.tol. To use with defaults just set restrain={} otherwise restrain =
    {'a': (float), 'b': (float), 'H': (bool), 'cycles': (int), 'tol': (float)}
    '''

    esp.set_parameters(restrain={})
    esp.execute()

    type = 'restrained'
    print(' {:10} '.format(type), end='')
    for i in esp.cache[type]:
        print('{:^12.4f}  '.format(i), end='')
    print('{:>8.4e}'.format(esp.cache['root mean square'][type]))

    '''
    It is possible to perform a final refinement both confined (or free) and restrained charges. This is will detect any
    carbon-hydrogen group and refit the charges to be the same and perform an additional restraint using 'a'=0.001 and
    'b'=0.1. All other atoms have their charges constrained to their restrained values. If 'restrain' has not been
    specified it will be assumed. To perform a refine use keyword 'refine={'apply': True}' adding optionally the dictionary
    keys 'a' and 'b' to change the default restraint fitting values. This molecule has no carbon groups so the 'refine'
    keyword would be ignored. The connectivity of the molecule used to determine the CH groups is found by using the
    is_bond(scf, atom1, atom2) utility in mol.utl - this uses the Van der Waals and the Mayer bond order to decide if a
    bond exists between atoms.
    '''

    '''
    The points at which the elctrostatic potential is sampled are available in the 'surface' property. This is a numpy array
    of dimension [number of point, 3] - the 1 axis being x,y,z. The electrostatic potential values at these points is in the
    property 'electrostatic_potential'
    '''

    '''
    there is a further method of distributing points on a sphere known as the 'Thomson problem'. This method asks the
    question 'how would electrons distribute themselves on the surface of a unit sphere under electrostatic forces'. Although
    this method is conceptually allied to rESP it is a lot slower than other methods/ If using this method an extra cache
    keyword 'thomson' is available which is a list of dictionaries detailing the Thomson analysis - please see wikipedia
    article for a discussion of the 'Thomson problem'.
    '''

    esp.set_parameters(distribution='thomson', density=1, count=1, scale=1.0)
    esp.execute()

    print('\n Thomson distribution')
    type = 'free' if not 'constrain' in esp.parameters else 'constrained'

    print('            ', end='')
    for i in esp.cache[type]:
        print('{:^12.4f}  '.format(i), end='')
    print('{:>8.4e}'.format(esp.cache['root mean square'][type]))

    print(' atom  shell    points   cycles       U[thomson]\n', '-'*50)
    for shell in esp.cache['thomson']:
        print('  {:2}   {:3}      {:4}      {:5} {:>18.10f}'.format(shell['atom']+1, shell['shell']+1, shell['points'],
                                                                shell['cycles'], shell['U']))

    '''
    Next we will look at the methylamine example...
    '''

    geo = (
'''
C
N   1   R1
H   1   H1  2   109.4712
H   1   H1  2   109.4712   3    120
H   1   H1  2   109.4712   3    240
H   2   H2  1   109.4712   3     60
H   2   H2  1   109.4712   3    300

R1  = 2.77788276200915
H1  = 2.05979061944896
H2  = 1.90861332627839
''')

    # Computation options
    mol = molecule( geo,
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=True)

    scf = RHF(mol,
              cycles=50,
              tol=1e-7,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    '''
    Initiate the RESP class by passing a RHF class object to the class initiator, this sets the default options
    '''
    esp = RESP(scf)

    '''
    set the parameters for the computation
    '''
    esp.set_parameters(distribution='connolly', density=1, count=4, increment=0.2, scale=1.4, restrain={}, refine={'apply':True, 'a':0.001} )

    '''
    It is possible to save the points, potential, inverse radii and parameters in a compressed numpy file. Use the keyword 'file' with
    file='w' to write the file (called rESP.npz in the application root directory). The computation can then be re-run by setting parameter
    file='r' before the execute(). Parameters setting from the saved run can be altered by subsequent set_parameters statements after the
    one declaring the file read. file='c' or file='rc' with delete the file after the current execute().
    '''
    esp.set_parameters(file = '')
    '''
    restraint parameters will be overwritten by refine so if you want them save them now
    '''
    restrain_parameters = esp.parameters['restrain'].copy()

    '''
    set Van der Waals radii to values in Harpy (GAMESS) for comparison with Harpy and psi3numpy. This implementation by default uses values
    in wikipedia and are same as used by PySCF
    '''

    esp.set_parameters(radii=[['C', 1.5],['N', 1.5], ['H', 1.2]])
    esp.execute()

    print('\nExample 2 - methylamine\n')

    print('Atomic shell construction\n-------------------------')
    print('   point distribution algorithm is ', esp.parameters['shell']['distribution'].capitalize())
    atom_types = sorted(list(set([atom.symbol for atom in scf.mol.atom])))
    h = atom_types.index('H') if atom_types.count('H') else -1
    if h != -1:
        del atom_types[h]
        atom_types += ['H']

    for i in atom_types:
        print('   atom species is [{:2}]   Van der Waals radius is [{:>5.2f}]  '.format(i, van_der_waals_radii[i]), end='')
        print('shells at radii ', end='')
        for j in range(esp.parameters['shell']['count']):
            print('{:>5.2f}'.format(van_der_waals_radii[i]*esp.parameters['shell']['scale']
                                    + esp.parameters['shell']['increment']*j), end='')
        print()
    if 'density' in esp.parameters['shell']:
        print('\n   Points are distributed at a density of {:>4.2f} points per square Angstrom'.format(esp.parameters['shell']['density']))
    if 'points' in esp.parameters['shell']:
        print('\n   Points are distributed at a frequency of {:>4.2f} points per shell'.format(esp.parameters['shell']['points']))

    print('   Total points ensemble is {:5} points'.format(esp.electrostatic_potential.shape[0]))


    print('\nElectrostatic Charges\n---------------------')

    constrained = 'constrained' if 'constrained' in esp.cache else 'not constrained'

    print('Charges are {:10}'.format(constrained.upper()))

    type = 'free' if constrained == 'not constrained' else 'constrained'
    for i in scf.mol.atom:
        print('{:^12}'.format(i.symbol+str(i.id+1)), end='')
    print('     \u0394rms')
    for i in esp.cache[type]:
        print('{:^12.4f}'.format(i), end='')
    print('  {:>8.4e}'.format(esp.cache['root mean square'][type]))

    restrained = 'restrained' if 'restrained' in esp.cache else 'not restrained'

    print('\nCharges are {:10}'. format(restrained.upper()))

    if restrained == 'restrained':
        print('\nRestraint parameters:-')
        print('    \'a\' is {:>6.4f} \n    \'b\' is {:>6.4f}'.format(restrain_parameters['a'], restrain_parameters['b']))
        print('    hydrogens are {:3} restrained'.format(['not', ''][restrain_parameters['H']]))
        print()
        for i in scf.mol.atom:
            print('{:^12}'.format(i.symbol+str(i.id+1)), end='')
        print('     \u0394rms')
        for i in esp.cache['restrained']:
            print('{:^12.4f}'.format(i), end='')
        print('  {:>8.4e}'.format(esp.cache['root mean square']['restrained']))

    carbon_groups = esp.carbon_groups

    if carbon_groups != [] and esp.parameters['refine']:

        print('\nCarbon group refinements\n------------------------')
        for group in carbon_groups:
            print('There is a CH grouping based on carbon [', group[0], '] and associated hydrogens ', group[1])

        print('\nRefined charges are constrained to:-')
        for x in esp.parameters['constrain']:
            if x[0] == '=':
                print('    Atoms ', x[1], 'are constrained to be equal charges')
            else:
                print('    Atom [{:2}] is constrained to {:>8.4f}'.format(x[1][0], x[0]))
        print()
        for i in scf.mol.atom:
            print('{:^12}'.format(i.symbol+str(i.id+1)), end='')
        print()
        for i in esp.cache['refined constrained']:
            print('{:^12.4f}'.format(i), end='')

        print()
        print('\nRefined charges are restrained with parameters:-')
        print('    \'a\' is {:>8.4e} \n    \'b\' is {:>6.4e}'.format(esp.parameters['refine']['a'], esp.parameters['refine']['b']))
        print()
        for i in scf.mol.atom:
            print('{:^12}'.format(i.symbol+str(i.id+1)), end='')
        print()
        for i in esp.cache['refined restrained']:
            print('{:^12.4f}'.format(i), end='')
        print()
    '''
    it is possible to get a visualization of the point distribution with the .view_surface() method. This has parameters elevation and
    azimuth, it's best to up the points count a stick to a single shell
    '''

    esp.set_parameters(distribution='connolly', density=40, count=1, scale=1)
    esp.execute()
    esp.view_surface(elevation=45, azimuth=45)
