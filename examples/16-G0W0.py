from __future__ import division
import numpy as np


from mol.mol import molecule
from scf.rhf import RHF
from phf.eig import solver
from phf.gwt import GW

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

    '''
    instatiate a solver object and pass instance to GW class. Create instance of GW class by passing rhf object
    , solver object and ask for 6 orbital from lowest occupied ie 1,2,3,4,homo,lumo in this case. 'analytic' method
    uses a Newton root finder with analytic derivative to find G0W0 energies. In the @HF case we are ignoring
    exchange interactions. A linear approximation is also calculated. These are returned in a cache property with
    keys 'koopman', 'analytic', 'linear', 'sigma' and 'poles'. The numeric method uses a 3-point central difference
    formula to calculate the derivatives of the self-energy in a linear approximation - cache keys 'koopman' and
    'numeric' are available. The graphic method does a numeric calculation on an extended linear energy grid and
    returns a cache with the key 'plot' which is a matplotlib plot object which can be displayed with the
    matplotlib.pyplot.show() method and will give subplots of the real part of the self-energy spectrum for the number
    of orbitals requested.
    '''
    solve = solver(roots=-1, vectors=True)
    gw = GW(scf, solve, orbital_count=6)
    gw.analytic()

    print('***************\n*   G0W0@HF   *\n***************')
    print('\nAnalytic derivatives')
    print(' orbital    Koopman      G0W0       G0W0-linear     pole        \u03A3\n', '-'*75)
    for orbital in range(gw.orbital_count):

        frontier = ''
        if orbital == gw.nocc: frontier = 'lumo'
        if orbital == (gw.nocc-1): frontier =  'homo'

        print('   {:2}     {:>10.4f} {:>10.4f}     {:>10.4f} {:>10.4f} {:>10.4f}   {:4}'.format(
              orbital, gw.cache['koopman'][orbital], gw.cache['analytic'][orbital],
              gw.cache['linear'][orbital], gw.cache['poles'][orbital], gw.cache['sigma'][orbital], frontier) )

    gw.numeric(grid_step=0.0001)
    print('\nNumeric derivatives')
    print(' orbital    Koopman   G0W0-linear\n', '-'*38)
    for orbital in range(gw.orbital_count):

        frontier = ''
        if orbital == gw.nocc: frontier = 'lumo'
        if orbital == (gw.nocc-1): frontier =  'homo'

        print('   {:2}     {:>10.4f} {:>10.4f}    {:4}'.format(
              orbital, gw.cache['koopman'][orbital], gw.cache['numeric'][orbital], frontier) )

    print('\nNumeric derivatives - linear grid self-energy plots...')
    gw.graphic(grid_step=0.01)
    gw.cache['plot'].show()
