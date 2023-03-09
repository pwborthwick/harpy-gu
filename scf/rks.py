from __future__ import division
import numpy as np

from scf.diis import dii_subspace
from scf.out import out
from int.aello import aello
from int.grd import GRID

from phf.eig import solver
from mol.mol import CONSTANTS

from scipy.linalg import fractional_matrix_power as fractPow

class RKS(object):

    def __init__(self, mol, mesh='fine', xc= 'LDA,VWN_RPA', cycles=50, tol=1e-6, diis=True, diis_size=6):

        self.mol = mol
        self.cycles = cycles
        self.tol=tol
        self.converged = False

        self.mesh = mesh
        self.xc = xc

        self.mom = self.get_orbital_occupation

        #do consistency checks
        if ((self.mol.spin + np.sum(self.mol.nele)) % 2) != 0:
            print('spin and number of electrons incompatible')
        self.open = (self.mol.spin != 0)

        if self.open:
            print('Incompatible spin for RKS')
            exit('use UKS')

        self.DIIS, self.diis_size = diis, diis_size

        #output details of molecule
        out(self.mol.silent, [self.mol, self.DIIS, self.diis_size, self.open, 'RKS', self.xc, self.mesh], 'initial')
        out(self.mol.silent, [cycles, tol], 'cycle')

        #basis information
        out(self.mol.silent, [self.mol], 'orbitals')

        #default is to converge to restricted solution
        self.closed_shell_behavior = 'r'

        self._cache = {}

    def get(self, key):
        #retrieve computed values

        return self._cache[key]

    def get_density_matrix(self, mo_coeff, mo_occ):
        #construct the one electron density matrix

        d =  np.einsum('pr,qr->pq', mo_coeff*mo_occ, mo_coeff, optimize=True)

        return np.array(d)

    def get_orbital_occupation(self, mo_energy, *_):
        #determine occupation numbers of orbitals

        #sort eigenvalues
        e_idx  = np.argsort(mo_energy)
        e_sort = mo_energy[e_idx]

        #doubly occupied orbitals - set to 1 the lowest occupied
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[e_idx[:self.mol.nele[0]]] = 1

        return mo_occ

    def execute(self):
        #main computation loop

        def compute_mo(f):
            #orthogonalise, solve and back-transform fock amtrix

            #orthogonalise Fock f->f' and solve
            fp = np.einsum('rp,rs,sq->pq', x, f, x, optimize=True )

            #use direct solver to diagonalise Fock
            solve.direct(fp)
            ep, cp = solve.values, solve.vectors

            #transform to ao basis
            c = np.einsum('pr,rq->pq', x, cp, optimize=True)

            return ep, c

        def get_energy(d, f):
            #compute the 1e, coulomb and exchange-correlation energies

            return np.einsum('pq,pq->', d, f, optimize=True)

        #get instance of solver object
        solve = solver(roots=-1, vectors=True)

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
        f = h_core

        mo_energy, mo_coeff = compute_mo(f)

        #get occupied coefficients
        mo_occupation = self.mom(mo_energy, mo_coeff, s)

        #density matrix
        d = set if type(set) == np.ndarray else self.get_density_matrix(mo_coeff, mo_occupation)

        #get grid
        grid, weights = GRID(self.mol.atom, self.mesh)

        #evaluate basis over grid
        if self.xc in ['LDA','VWN_RPA','LDA,VWN_RPA']: from int.ksn import LDA as DFT


        dft = DFT(self.mol.orbital, self.xc)
        ao = dft.evaluate_atomic_orbital(grid)

        last_cycle_energy = 0.0

        #diis initialisation
        if self.DIIS: diis = dii_subspace(self.diis_size)

        extra_cycle = False
        for cycle in range(1, self.cycles):

            #build the coulomb integral
            j = 2.0 * np.einsum('rs,pqrs->pq', d, eri, optimize=True)

            #evalute density over mesh
            rho = dft.evaluate_rho(d, ao)

            #evaluate functional over mesh
            exc, vxc = dft.functional(rho)

            #evaluate potential
            vxc = dft.evaluate_vxc(vxc, ao, weights)

            f = h_core + j + vxc

            if (cycle != 0) and self.DIIS:
                f = diis.build(f, d, s, x)

            mo_energy, mo_coeff = compute_mo(f)

            #get occupied coefficients
            mo_occupation = self.mom(mo_energy, mo_coeff, s)

            d = self.get_density_matrix(mo_coeff, mo_occupation)

            eSCF = get_energy(d, 2.0*h_core + j) + dft.evaluate_exc(exc, rho, weights)

            vector_norm = diis.norm if self.DIIS else ''
            delta_energy = abs(eSCF - last_cycle_energy)

            #check energy level behaviour
            homo = self.mol.nele[0]-1
            warn = '** homo = lumo' if mo_energy[homo] + 0.001 > mo_energy[homo+1] else ''

            out(self.mol.silent, [cycle, np.einsum('pq,pq->', d, (2.0*h_core), optimize=True),
                np.einsum('pq,pq->', d, ( j), optimize=True),
                dft.evaluate_exc(exc, rho, weights),
                np.sum(rho*weights),  delta_energy, vector_norm, warn],'rks')

            if  delta_energy < self.tol:
                self.converged = True
                if extra_cycle: break
                extra_cycle = not extra_cycle


            last_cycle_energy = eSCF

        #final energies
        out(self.mol.silent, [eSCF, self.mol.nuclear_repulsion() ], 'final')

        #load cache with computed values
        self._cache['s'] = s ; self._cache['v'] = v ; self._cache['t'] = t ; self._cache['i'] = eri
        self._cache['f'] = f ; self._cache['d'] = d ; self._cache['c'] = mo_coeff ; self._cache['e'] = mo_energy ;
        self._cache['o'] = mo_occupation ; self._cache['x'] = vxc ; self._cache['g'] = grid ; self._cache['w'] = weights

        total_energy = eSCF + self.mol.nuclear_repulsion()

        return total_energy

    def analyse(self, method=''):
        #simple post HF

        types = method.split(',')
        for type in types:
            if type == 'dipole':

                debyes = CONSTANTS('au->debye')

                dipole = np.array(aello(self.mol.atom, self.mol.orbital, 'dipole', None, self.mol.gauge))

                mu = -np.einsum('xii->x',np.einsum('pr,xrq->xpq', 2.0*self.get('d'), dipole, optimize=True))
                #nuclear component and charge center adjustment
                for i in range(3):
                    for j in range(self.mol.natm):
                        mu[i] += self.mol.atom[j].number * (self.mol.atom[j].center[i] - self.mol.gauge[i])

                mu *= debyes

                self.dipole = mu
                out(self.mol.silent, mu, 'rdipole')

            if type == 'geometry':
                out(self.mol.silent, self.mol, 'geometry')

            if type == 'charge':

                u = fractPow(self.get('s'), 0.5).real

                population = np.zeros(self.mol.natm)
                pop = 2.0 * np.einsum('ij,jk,ki->i',u, self.get('d'), u, optimize=True)

                for i, a in enumerate(self.mol.atom):
                    for j, o in enumerate(self.mol.orbital):
                        if o.atom.id == i: population[i] += pop[j]

                charge = np.array([a.number - population[i] for i, a in enumerate(self.mol.atom)])

                self.charge = charge
                out(self.mol.silent, [population, charge], 'rcharge')

            if type == 'bonds':

                mayer = 2.0 * np.einsum('pr,rq->pq', self.get('d'), self.get('s'), optimize=True)

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

