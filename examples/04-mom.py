from __future__ import division
from mol.mol import molecule
from scf.uhf import UHF
from mol.mol import CONSTANTS

if __name__ == '__main__':
    mol = molecule([['O', (0.0, 0.0, 0.0)], 
                    ['H', (0,-0.757 ,0.587)], 
                    ['H', (0, 0.757 ,0.587)]], 
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='3-21g',
                    silent=False)

    scf = UHF(mol, 
                cycles=50,
                tol=1e-8,
                diis=True)

#***************************************
#* Maximum Overlap Method - Excitation *
#***************************************

    print('Ground State Computation\n------------------------')
    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    ground_state_energy = scf_energy

    #ground state properties
    ph_coeff = scf.get('c')
    ph_occ   = scf.get('o')

    #excitation occupancy homo->lumo
    ph_occ[0][4] = 0.0
    ph_occ[0][5] = 1.0

    #computation with new occupancy
    ph_scf = UHF(mol,
                        cycles=50,
                        tol=1e-8,
                        diis=True)

    ph_scf.maximum_overlap_method(ph_coeff, ph_occ)

    print('\nExcited State Computation\n------------------------')
    ph_energy = ph_scf.execute()
    if not ph_scf.converged: exit('SCF convergence failed')

    energy_gap = (ph_energy - ground_state_energy) * CONSTANTS('hartree->eV')
    print('\n---------->Excitation Energy (HOMO->LUMO) {:<6.2f} eV'.format(energy_gap))

    #Koopman's theorem states IP is -HOMO energy
    homo = scf.get('e')[0][scf.mol.nele[0]-1]
    print('\n---------->Koopmans - Energy of HOMO {:<6.2f} eV'.format(-homo * CONSTANTS('hartree->eV')))

    '''
    Ground State Computation
    ------------------------

    nuclear repulsion      9.1882584177
    total electronic     -84.7736598019

    final total energy   -75.5854013841

    Excited State Computation
    ------------------------

    nuclear repulsion      9.1882584177
    total electronic     -84.4898762240

    final total energy   -75.3016178063

    ---------->Excitation Energy (HOMO->LUMO) 7.72   eV

    ---------->Koopmans - Energy of HOMO 13.05  eV
    '''

    #The delta SCF method for IP/EA relies on the difference between the ground state energy and the energy of the same geometry with
    #(n1-) or (n+1) electrons eg the energy of the above system with 9 electrons is
    mol = molecule([['O', (0.0, 0.0, 0.0)], 
                    ['H', (0,-0.757 ,0.587)], 
                    ['H', (0, 0.757 ,0.587)]], 
                    spin=1,
                    units='angstrom',
                    charge=1,
                    gto='3-21g',
                    silent=False)

    scf = UHF(mol, 
                cycles=50,
                tol=1e-8,
                diis=True)

    anion_energy = scf.execute()
    if scf.converged:
        print('Delta SCF method\n----------------')
        print('---------->Delta SCF IP {:<6.3} eV'.format((anion_energy - ground_state_energy) * CONSTANTS('hartree->eV')))

    '''
    Delta SCF method
    ----------------
    ---------->Delta SCF IP 10.5   eV

    '''

    #The 'best' metrhod is the Electron Propogator (3) - run as RHF
    from scf.rhf import RHF
    from phf.ept import EP
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
                tol=1e-8,
                diis=True)

    rhf_energy = scf.execute()
    if scf.converged:
        ep = EP(scf, orbital_range='HOMO,HOMO')
        ep.method(order=3)
        print('---------->Electron Propagator (3) {:<6.3f} eV'.format(-ep.cache[0]['sigma'] * CONSTANTS('hartree->eV')))

        ep.method(order=2)
        print('---------->Electron Propagator (2) {:<6.3f} eV'.format(-ep.cache[0]['sigma'] * CONSTANTS('hartree->eV')))


    '''
    ---------->Electron Propagator (3) 11.538 eV
    ---------->Electron Propagator (2) 10.514 eV
    '''
