from __future__ import division
import numpy as np
import scipy as sp
from scipy.linalg import fractional_matrix_power as fractPow

from int.aello import aello
from scf.diis import dii_subspace
from scf.out import out
from mol.mol import molecule

'''
ROHF Theory Made Simple - [https://arxiv.org/pdf/1008.1607.pdf]
gatech - [http://vergil.chemistry.gatech.edu/notes/cis/node5.html]
'''

class ROHF(object):

    def __init__(self, mol, cycles=50, tol=1e-6, diis=True, diis_size=6):

        self.mol = mol
        self.cycles = cycles
        self.tol=tol
        self.converged = False

        #do consistency checks
        if ((self.mol.spin + np.sum(self.mol.nele)) % 2) != 0:
            print('spin and number of electrons incompatible')
        self.open = (self.mol.spin != 0)

        self.DIIS = diis ; self.diis_size = diis_size

        #output details of molecule
        out(self.mol.silent, [self.mol, self.DIIS, self.diis_size, self.open, 'ROHF'], 'initial')
        out(self.mol.silent, [cycles, tol], 'cycle')

        self._cache = {}

    def get(self, key):
        #retrieve computed values

        return self._cache[key]

    def get_spin_statistics(self, mo_coeff, mo_occ, s):
        #compute the actual spin squared and multiplicity

        alpha, beta = (mo_occ > 0).astype(np.int), (mo_occ == 2).astype(np.int)

        #get occupied molecular eigenvectors
        occupation = [mo_coeff*alpha, mo_coeff*beta]
        mo_occ = [occupation[0][:,~np.all(occupation[0]==0, axis=0)],  occupation[1][:,~np.all(occupation[1]==0, axis=0)]]
        n_alpha, n_beta = [mo_occ[0].shape[1], mo_occ[1].shape[1]]

        #components of total spin
        s = np.einsum('rp,rs,sq->pq',mo_occ[0], s, mo_occ[1], optimize=True)

        spin = []
        spin.append((n_alpha + n_beta) * 0.5 - np.einsum('ij,ij->', s, s, optimize=True))
        spin.append(pow(n_alpha - n_beta, 2) * 0.25)

        return sum(spin), 2 * np.sqrt(sum(spin) + 0.25)

    def get_density_matrix(self, mo_coeff, mo_occ):
        #construct the one electron density matrix

        core = [i != 0 for i in mo_occ]
        spin = [i == 2 for i in mo_occ]

        da =  np.einsum('pr,r,qr->pq', mo_coeff, core, mo_coeff, optimize=True)
        db  = np.einsum('pr,r,qr->pq', mo_coeff, spin, mo_coeff, optimize=True)

        return np.array([da, db])

    def build_fock(self, h_core, jk, d, s):
        #build the spin and Roothaan Fock matrices

        #UHF fock operators
        uhf_f = (h_core + jk[0], h_core + jk[1])

        f_core = (uhf_f[0] + uhf_f[1]) * 0.5

        #project UHF Focks - alpha all non-zero, beta all doubly occupied
        core = np.einsum('pr,rq->pq', d[1], s, optimize=True)
        spin = np.einsum('pr,rq->pq', d[0] - d[1], s, optimize=True)
        virt = np.eye(self.mol.norb) - np.einsum('pr,rq->pq', d[0], s, optimize=True)

        #build Fock
        f  = 0.5 * np.einsum('sp,sr,rq->pq', core, f_core, core, optimize=True)
        f += 0.5 * np.einsum('sp,sr,rq->pq', spin, f_core, spin, optimize=True)
        f += 0.5 * np.einsum('sp,sr,rq->pq', virt, f_core, virt, optimize=True)

        f += np.einsum('sp,sr,rq->pq', spin, uhf_f[1], core, optimize=True)
        f += np.einsum('sp,sr,rq->pq', spin, uhf_f[0], virt, optimize=True)
        f += np.einsum('sp,sr,rq->pq', virt, f_core, core, optimize=True)

        #Roothan Fock
        roothaan_fock = f + f.transpose(1,0)

        return uhf_f, roothaan_fock

    def get_orbital_occupation(self, mo_energy):
        #determine the electrons in each orbital

        #get orbital energies
        size = len(mo_energy)
        if size == 1:
            eps = mo_energy[0]
            e_alpha = e_beta = eps
        else:
            eps, e_alpha, e_beta = mo_energy

        #occupation numbers
        occupations = np.zeros(self.mol.norb)
        eps_sorted = np.argsort(eps)

        core = eps_sorted[:self.mol.nele[0]]

        if self.mol.spin != 0:
            spin = eps_sorted[self.mol.nele[0]:]
            spin = spin[np.argsort(e_alpha[spin])[:self.mol.spin]]
            occupations[spin] = 1

        occupations[core] = 2

        return occupations

    def execute(self, set=None):

        def compute_mo(f_spin, f_roothaan, x):
            #get the molecular coefficients fro the Fock

            #solve eigensystem
            p_fock = np.einsum('rp,rs,sq->pq', x, f_roothaan, x, optimize=True)
            e , c  = np.linalg.eigh(p_fock)
            mo_coeff =  np.einsum('pr,rq->pq',x, c, optimize=True)

            e_spin = np.einsum('pi,xps,si->xi', mo_coeff, f_spin, mo_coeff, optimize=True)

            return e, e_spin, mo_coeff

        def get_energy(d, h_core, eri, cycle=-1):
            #compute the 1e, coulomb and exchange-correlation energies

            #coulomb and exchange
            j = np.einsum('ijkl,xji->xkl', eri, d, optimize=True)
            k = np.einsum('ikjl,xji->xkl', eri, d, optimize=True)

            jk = j[0] + j[1] - k if cycle != 0 else j - k

            e =  np.einsum('ij,xji->', h_core, d, optimize=True)
            e_coulomb = (np.einsum('ij,ji->', jk[0], d[0]) + np.einsum('ij,ji->', jk[1], d[1])) * 0.5

            return (e, e_coulomb), jk

        #use a reduced version of Harpy's cython integrals
        s, t, v, eri = aello(self.mol.atom, self.mol.orbital)

        #orthogonal transformation matrix
        x = fractPow(s, -0.5).real

        #initial fock is core hamiltonian
        h_core = t + v

        #get core and open electron counts
        core_electrons = (np.sum(self.mol.nele) - self.mol.spin)//2
        self.mol.nele = [core_electrons, self.mol.spin]

        #orbital occupation
        eps, mo_coeff = sp.linalg.eigh(h_core, s)
        mo_occupation = self.get_orbital_occupation([eps])

        #get density
        d = set if type(set) == np.array else self.get_density_matrix(mo_coeff, mo_occupation)

        #ROHF energy components
        eSCF, jk = get_energy(d, h_core, eri, cycle=0)

        last_cycle_energy = 0.0

        #diis initialisation
        if self.DIIS: diis = dii_subspace(self.diis_size)

        extra_cycle = False
        self.cycles = 20
        for cycle in range(1, self.cycles):

            #construct Fock matrices
            f_spin, f_roothaan = self.build_fock(h_core, jk, d, s)

            if (cycle != 0) and self.DIIS:
                f_roothaan = diis.build(f_roothaan, d[0] + d[1], s, x)

            #solve for molecular orbital coefficients
            mo_energy, mo_spin, mo_coeff = compute_mo(f_spin, f_roothaan, x)

            mo_occupation = self.get_orbital_occupation([mo_energy, mo_spin[0], mo_spin[1]])

            d = self.get_density_matrix(mo_coeff, mo_occupation)

            #recalculate jk
            eSCF, jk = get_energy(d, h_core, eri)

            vector_norm = diis.norm if self.DIIS else ''

            out(self.mol.silent, [cycle, sum(eSCF), abs(sum(eSCF) - last_cycle_energy), vector_norm,
                self.get_spin_statistics(mo_coeff, mo_occupation, s), ''], 'uhf')

            if abs(sum(eSCF) - last_cycle_energy) < self.tol:
                self.converged = True
                self.reference = [sum(eSCF), self.mol.nuclear_repulsion()]
                if extra_cycle: break
                extra_cycle = not extra_cycle

            last_cycle_energy = sum(eSCF)
        else|:|
            return None

        #final energies
        out(self.mol.silent, self.reference, 'final')

        #load cache with computed values
        self._cache['s'] = s ; self._cache['v'] = v ; self._cache['t'] = t ; self._cache['i'] = eri
        self._cache['f'] = f_roothaan ; self._cache['d'] = d ; self._cache['c'] = mo_coeff
        self._cache['e'] = mo_energy ; self._cache['o'] = mo_occupation

        total_energy = sum(eSCF) + self.mol.nuclear_repulsion()

        #restore class variable nele
        self.mol.nele = sum([i.number for i in self.mol.atom]) - self.mol.charge

        return total_energy

    from scf.uhf import UHF
    analyse = UHF.analyse
