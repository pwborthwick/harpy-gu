from __future__ import division
import numpy as np
from scipy.linalg import fractional_matrix_power as fractPow

from int.aello import aello
from scf.diis import dii_subspace
from scf.out import out
from mol.mol import CONSTANTS

'''
UHF - [http://vergil.chemistry.gatech.edu/notes/cis/node4.html]
Maximum Overlap Method - [https://pubs.acs.org/doi/10.1021/jp801738f]
Breaking spatial-symmetry - [https://chemistry.stackexchange.com/questions/42005/initial-guess-for-unrestricted-hartree-fock-calculation]
'''
class UHF(object):

    def __init__(self, mol, cycles=50, tol=1e-6, diis=True, diis_size=6):

        self.mol = mol
        self.cycles = cycles
        self.tol=tol
        self.mom = self.get_orbital_occupation
        self.converged = False

        #do consistency checks
        if ((self.mol.spin + np.sum(self.mol.nele)) % 2) != 0:
            print('spin and number of electrons incompatible')
        self.open = (self.mol.spin != 0)

        self.DIIS = diis ; self.diis_size = diis_size

        #default is to converge to restricted solution
        self.closed_shell_behavior = 'r'

        #output details of molecule
        out(self.mol.silent, [self.mol, self.DIIS, self.diis_size, self.open, 'UHF'], 'initial')
        out(self.mol.silent, [cycles, tol], 'cycle')

        #basis information
        out(self.mol.silent, [self.mol], 'orbitals')

        self._cache = {}

    def get(self, key):
        #retrieve computed values

        return self._cache[key]

    def closed_shell(self, mo_coeff):
        #break symmetry for unrestriced closed-shell
        from math import cos, sin

        #rotate homo and lumo by 45 degrees
        homo = self.mol.nele[0]
        theta = 0.25 * np.pi

        #rotation matrix
        rotate = np.array([[cos(theta), -sin(theta)],
                           [sin(theta),  cos(theta)]])

        #rotate homo and lumo
        c = mo_coeff[0][:, homo-1:homo+1]
        mo_coeff[0][:,homo-1:homo+1] = np.einsum('pr,qr->qp',  rotate, c, optimize=True)

        return mo_coeff

    def get_density_matrix(self, mo_coeff, mo_occ):
        #construct the one electron density matrix

        da =  np.einsum('pr,qr->pq', mo_coeff[0]*mo_occ[0], mo_coeff[0], optimize=True)
        db  = np.einsum('pr,qr->pq', mo_coeff[1]*mo_occ[1], mo_coeff[1], optimize=True)

        return np.array([da, db])

    def get_spin_statistics(self, mo_coeff, mo_occ, s):
        #compute the actual spin squared and multiplicity

        alpha, beta = (mo_coeff[0][:, mo_occ[0]>0], mo_coeff[1][:,mo_occ[1]>0])
        occupations = (alpha.shape[1], beta.shape[1])

        s = np.einsum('rp,rs,sq->pq', alpha, s, beta, optimize=True)

        #get spin components [xy] and [z]
        spin = []
        spin.append(sum(occupations) * 0.5 - np.einsum('ij,ij->', s, s))
        spin.append(pow(occupations[1] - occupations[0], 2) * 0.25)
        ss = sum(spin)

        return ss, 2 * np.sqrt(ss + 0.25)

    def get_orbital_occupation(self, mo_energy, *_):
        #determine occupation numbers of orbitals

        #sort eigenvalues
        e_idx  = (np.argsort(mo_energy[0]), np.argsort(mo_energy[1]))
        e_sort = (mo_energy[0][e_idx[0]], mo_energy[1][e_idx[1]])

        #occupied orbitals - set to 1 the lowest occupied
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[0, e_idx[0][:self.mol.nele[0]]] = 1 ;  mo_occ[1, e_idx[1][:self.mol.nele[1]]] = 1

        return mo_occ

    def maximum_overlap_method(self, occorb, setocc):
        #The Maximum Overlap Method

        imposed_occupation = (occorb[0][:, setocc[0]>0] , occorb[1][:, setocc[1]>0])

        def get_orbital_occupation(_ , mo_coeff, s):
            #Use the MoM to generate next generation of occupation numbers

            mo_occ = np.zeros_like(setocc)

            for spin in [0, 1]:

                #compute the overlap between old and new coefficients
                occupation_number = int(np.sum(setocc[spin]))
                mom_s = np.einsum('rp,rs,sq->pq', imposed_occupation[spin], s, mo_coeff[spin], optimize=True)

                #get maximum overlap
                idx = np.argsort(np.einsum('ij,ij->j', mom_s, mom_s))[::-1]
                mo_occ[spin][idx[:occupation_number]] = 1

            return mo_occ

        self.mom = get_orbital_occupation


    def execute(self, set=None):
        #main computation loop

        def compute_mo(f):
            #orthogonalise, solve and back-transform fock amtrix

            #orthogonalise Fock f->f' and solve
            fp = np.einsum('rp,xrs,sq->xpq', x, f, x, optimize=True )
            ep , cp = np.linalg.eigh(fp)

            #transform to ao basis
            c = np.einsum('pr,xrq->xpq', x, cp, optimize=True)

            return ep, c

        def get_energy(d, core_h, f, cycle=0):
            #compute the 1e, coulomb and exchange-correlation energies

            eSCF = 0.5 * np.einsum('xpq,xpq->', d, (core_h + f), optimize=True)

            return eSCF

        #use a reduced version of Harpy's cython integrals
        s, t, v, eri = aello(self.mol.atom, self.mol.orbital)

        #orthogonal transformation matrix
        x = fractPow(s, -0.5).real

        #initial fock is core hamiltonian
        h_core = t + v

        #get alpha and beta electron counts
        paired_electrons = (np.sum(self.mol.nele) - self.mol.spin)//2
        self.mol.nele = [paired_electrons+self.mol.spin, paired_electrons]

        #initial Fock guess
        f = (h_core, h_core)

        mo_energy, mo_coeff = compute_mo(f)

        #get occupied coefficients
        mo_occupation = self.mom(mo_energy, mo_coeff, s)

        #unrestricted handler for closed closed shell
        if (not self.open) and self.closed_shell_behavior != 'r' :
            mo_coeff = self.closed_shell(mo_coeff)

        #alpha and beta density matrix
        d = set if type(set) == np.ndarray else self.get_density_matrix(mo_coeff, mo_occupation)

        eSCF = get_energy(d, h_core, f)

        last_cycle_energy = 0.0

        #diis initialisation
        if self.DIIS: diis = dii_subspace(self.diis_size)

        extra_cycle = False
        for cycle in range(1, self.cycles):

            #construct Fock matrices
            f = h_core + np.einsum('xrs,pqrs->xpq', d, eri, optimize=True)
            f -= np.einsum('xrs,psrq->xpq', d, eri, optimize=True)
            f += np.einsum('xrs,pqrs->xpq', (d[1], d[0]), eri, optimize=True)

            if (cycle != 0) and self.DIIS:
                f = diis.build(f, d, s, x)

            mo_energy, mo_coeff = compute_mo(f)

            #get occupied coefficients
            mo_occupation = self.mom(mo_energy, mo_coeff, s)

            d = self.get_density_matrix(mo_coeff, mo_occupation)

            eSCF = get_energy(d, h_core, f, cycle)

            vector_norm = diis.norm if self.DIIS else ''

            #check energy level behaviour
            homo = self.mol.nele[0]-1
            lumo = min(homo+1, len(mo_energy[0])-1)

            warn = '** homo = lumo' if mo_energy[0][homo] + 0.001 > mo_energy[0][lumo] else ''

            out(self.mol.silent, [cycle, eSCF, abs(eSCF - last_cycle_energy), vector_norm,
                self.get_spin_statistics(mo_coeff, mo_occupation, s), warn], 'uhf')

            if abs(eSCF - last_cycle_energy) < self.tol:
                self.converged = True
                self.reference = [eSCF, self.mol.nuclear_repulsion()]
                if extra_cycle: break
                extra_cycle = not extra_cycle


            last_cycle_energy = eSCF
        else:
            return None

        #final energies
        out(self.mol.silent, self.reference, 'final')


        #load cache with computed values
        self._cache['s'] = s ; self._cache['v'] = v ; self._cache['t'] = t ; self._cache['i'] = eri
        self._cache['f'] = f ; self._cache['d'] = d ; self._cache['c'] = mo_coeff ; self._cache['e'] = mo_energy ; self._cache['o'] = mo_occupation

        total_energy = eSCF + self.mol.nuclear_repulsion()

        return total_energy

    def analyse(self, method=''):
        #simple post HF

        types = method.split(',')
        for type in types:
            if type == 'dipole':

                debyes = CONSTANTS('au->debye')

                d = self.get('d')
                dipole = np.array(aello(self.mol.atom, self.mol.orbital, 'dipole', None, self.mol.gauge))

                mu = -np.einsum('Xxii->Xx',np.einsum('Xpr,xrq->Xxpq', d, dipole, optimize=True))

                #nuclear component and charge center adjustment
                for i in range(3):
                    nuclear_dipole = 0.0
                    for j in range(self.mol.natm):
                        nuclear_dipole +=  self.mol.atom[j].number * (self.mol.atom[j].center[i] - self.mol.gauge[i])

                    mu[0,i] += nuclear_dipole * 0.5
                    mu[1,i] += nuclear_dipole * 0.5

                mu *= debyes

                self.dipole = mu
                out(self.mol.silent, mu, 'udipole')

            if type == 'geometry':
                out(self.mol.silent, self.mol, 'geometry')

            if type == 'charge':

                u = fractPow(self.get('s'), 0.5).real

                population = np.zeros((2, self.mol.natm))
                pop = np.einsum('ij,xjk,ki->xi',u, self.get('d'), u, optimize=True)

                for i, a in enumerate(self.mol.atom):
                    for j, o in enumerate(self.mol.orbital):
                        if o.atom.id == i: population[:,i] += pop[:,j]

                charge = np.array([a.number - sum(population[:,i]) for i, a in enumerate(self.mol.atom)])
                self.charge = charge
                out(self.mol.silent, [population, charge], 'ucharge')

            if type == 'bonds':

                d = self.get('d')[0] + self.get('d')[1]

                mayer = np.einsum('pr,rq->pq', d, self.get('s'), optimize=True)

                bond_order = np.zeros((self.mol.natm, self.mol.natm))
                valence = np.zeros(self.mol.natm)

                for i, p in enumerate(self.mol.orbital):
                    for j in range(i):

                        mu = p.atom.id
                        nu = self.mol.orbital[j].atom.id

                        if mu != nu: bond_order[mu, nu] += mayer[i, j] * mayer[j, i]

                bond_order += bond_order.transpose()
                for i in range(self.mol.natm):
                    for j in range(self.mol.natm):

                        valence[i] += bond_order[i,j]

                out(self.mol.silent, [bond_order, valence, self.mol], 'bonds')
