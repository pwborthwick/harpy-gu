from mol.mol import molecule
from scf.rhf import RHF
from scf.uhf import UHF

import contextlib
with contextlib.redirect_stdout(None):
    print('here')

if __name__ == '__main__':
    mol = molecule([['H', (0.0 , 0.0, 0.0)], 
                    ['H', (1.39838996182773, 0.0 ,0.0)]], 
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=True)

    #restricted Hartree-Fock class object
    rhf = RHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=False)
    '''
    solve reference geometry so can get <00||00> which is dissociated energy
    '''
    rhf.execute()
    coulomb_repulsion = rhf.get('i')[0,0,0,0] * 0.5
    
    #unrestricted Hartree-Fock class object
    uhf = UHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=False)
    uhf.closed_shell_behavior = ''

    '''
    Hydrogen atom 1 is positioned at the origin while the second hydrogen is moved along the positive x-axis
    starting separation in 0.25 anstroms and we move to 6 amgstroms in steps of 0.25 angstrom
    '''
    rhf_e = [] ; uhf_e = []
    r = []
    for x in range(100):
        r.append(0.5 + x*0.125)
  
        '''
        update the geometry
        '''
        centers = [r[-1], 0.0, 0.0]
        mol.atom[1].center = centers

        rhf_energy = rhf.execute()
        rhf_e.append(rhf_energy)


        uhf.closed_shell_behavior = 'u'
        uhf_energy = uhf.execute()
        uhf_e.append(uhf_energy)

'''
single Hydrogen atom energy
'''
mol.atom = mol.atom[:1]
mol.orbital = mol.orbital[:1]
uhf.mol.natm = 1
mol.spin = 1 
uhf.closed_shell_behavior = 'r'
H = uhf.execute()

import matplotlib.pyplot as plt
plt.plot(r, rhf_e, '.r')
plt.plot(r, uhf_e, '.m')

plt.plot([0,13], [2*H, 2*H], 'k.--')
plt.plot([0,13], [2*H + coulomb_repulsion,2*H + coulomb_repulsion], 'k.--')

plt.text(12, -0.65, 'rhf')
plt.text(12, -1.00, 'uhf')
plt.text(1.8,-0.52,'2H + ' + r'$\frac{1}{2} [00 \vert 00]$', size='x-small')
plt.text(1.8,-0.91,'2H', size='x-small')

plt.xlabel('bond length (bohr)')
plt.ylabel('energy (Hartree)')

plt.title('H$_2$' + ' dissociation rhf v uhf')

plt.show()

'''
At bonding separation the model is H1 + <11|22> + H2 or (H1 + 0.5<11|22>) + (0.5<11|22> + H2). At large separation in the RHF case there are two hydrogen atom energies, 
but the two electrons must still share a doubly occupied spatial orbital as a bonding orbital doesn't exist at large separation the two electrons are forced into an
orbital on just one of the hydrogens. This means the model is essentially H^+, H^- and the energy of half a doubly occupied 2-electron repulsion integral. This level is
shown as the upper horizonatal line in the plot. In the UHF case each electron can stay with its nucleus in a singly occupied spin orbitl leaving a H - H model, the H2 
energy is shown in the plot as the lower horizonatal line.
'''

