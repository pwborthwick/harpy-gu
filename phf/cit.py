from __future__ import division
import numpy as np
import scipy as sp

from itertools import combinations
from math import factorial

import int.mo_spin as mos
from phf.eig import solver
from int.aello import aello
from phf.fci import FCI as mFCI

'''
CIS and RPA - Crawford projects 12 - [https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2312]
FCI - [https://hal.archives-ouvertes.fr/hal-01539072/document] [https://tel.archives-ouvertes.fr/tel-02089570v1/document]
CIS_MP2 and CIS(D) - [https://hal.archives-ouvertes.fr/hal-02879256/document]
RPA and CIS - [https://joshuagoings.com/2013/05/27/tdhf-cis-in-python/]
Chapter 4 - Modern Quantum Chemistry by A Szabo and N.S. Ostlund
'''

class CI(object):
    #Configuration Interaction class

    def __init__(self, scf):

        self.scf = scf
        self.davidson = False

        self.transition_method = ''

        global nmo, ndocc
        nmo, ndocc = self.scf.mol.norb, self.scf.mol.nele[0]

    def get_fci_dominant(self, number):
        #get the first 'number' largest elements of Hamiltonian and determinants

        #subtract SCF ground state energy
        ndet, nmo = self.hamiltonian.shape[0], self.scf.mol.norb
        h = (self.hamiltonian - (np.eye(ndet) * self.scf.reference[0])).flatten()

        #get indices of largest absolute elements
        idx = np.argpartition(np.abs(h), -number)[-number:]
        idx = idx[np.argsort(-np.abs(h)[idx])]
        idx = np.unravel_index(idx, h.shape)

        #retrieve the values of elements
        val = h[idx]

        #get pointer to determinants
        dets = [[i//ndet, i - (i//ndet) * ndet] for i in idx][0]

        data = [[val[i], self.determinant_list[dets[0][i]], self.determinant_list[dets[1][i]]] for i in range(number)]

        return data

    def degeneracies(self, vector):
        #determine the degeneracies of a vector

        vector = vector[:]

        degeneracy_tuples = []
        while len(vector) != 0:

            #get the first element to test all other values against
            test = vector[0]

            #mark as True all matches and count True values
            matches = np.isclose(vector, test)
            counts = np.count_nonzero(matches == True)

            #remove all matches from vector
            idx = [n for n, i in enumerate(matches) if i]
            vector = np.delete(vector, idx)

            degeneracy_tuples.append((test, counts))

        return degeneracy_tuples

    def FCI(self, code='S', spin_adapt=False, use_residues=''):
        '''
        Full spin Configuration Interaction
        ! this is a shell around the FCI module imported as mFCI !
        this modules provides for spin-adapted determinant bases and to generate the 
        singles and doubles residues
        '''

        nmo, nocc = self.scf.mol.norb, self.scf.mol.nele[0]

        mfci = mFCI(self.scf, method=code, spin_adapt=spin_adapt, use_residues=use_residues)

        self.hamiltonian = mfci.hamiltonian
        self.determinant_list = mfci.determinant_list

        info = str(mfci.__str__()).split(',')
        #properties and determinant count statistics
        self.statistics = {'requested level': info[0], 'method': info[1], 
                           'FCI determinant space': info[2],
                           'effective space': info[3]}

        if spin_adapt: self.is_singlet = mfci.is_singlet

        if code == 'S' and not spin_adapt: self.transition_method = 'fci'

    def CIS(self):
        #simple CIS

        #orbital occupations and orbital slices
        nocc = sum(self.scf.mol.nele)
        nvir = self.scf.mol.norb*2 - nocc                                    
        o, v = slice(None, nocc), slice(nocc, None)                           

        #get mo spin-basis 2-electron integrals and Fock
        g = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))          
        f = mos.orbital_transform(self.scf, 'm+s', self.scf.get('f'))

        #compute the A and B excitation and de-excitation matrices
        ones = np.ones(nocc + nvir)
        h =  np.einsum('ab,ij->iajb',np.diag(np.diag(f)[v]),np.diag(ones[o]))
        h -= np.einsum('ij,ab->iajb',np.diag(np.diag(f)[o]),np.diag(ones[v]))
        h += g[v, o, o, v].transpose(2,0,1,3)

        self.hamiltonian = h.reshape(nocc*nvir, nocc*nvir)

        #Davidson methods
        self.davidson = True
        ds = mos.orbital_deltas(self.scf, 1, 's')[0]

        def diagonal(self):
            #CIS diagonal for Davidson iterations

            #initialize to fock diagonal
            diagonal = -ds.ravel()

            cis_diagonal = diagonal[:(nocc*nvir)].reshape(nocc, nvir)
            cis_diagonal -= np.einsum('aiai->ia', g[v, o, v, o], optimize=True)

            return diagonal

        self.diagonal = diagonal

        def guess(self, roots, f=1):
            #initial vector to start Davidson

            diag = self.diagonal(self)

            #get largest absolute values on diagonal matrix as best guess
            args = np.argsort(np.absolute(diag))

            #we only have nocc*nvir roots available
            if roots > len(args):
                print('reducing requested roots - exceeded ', len(args))
                roots = len(args)

            guess_vectors = np.zeros((diag.size, roots * f))
            for root in range(roots * f):
                guess_vectors[args[root], root] = 1.0

            return np.array(guess_vectors)

        self.guess = guess

        def matvec(self, cis):
            #construct the blocks of CIS matrix dot product with arbitary vector (r)

            cis    = np.array(cis)
            sigma  = np.zeros_like(cis)

            cis_sigma = cis[:(nocc*nvir)].reshape(nocc, nvir)
            sigma_s  = sigma[:(nocc*nvir)].reshape(nocc, nvir)

            #singles sigmas
            sigma_s -= np.einsum('ia,ia->ia', ds, cis_sigma, optimize=True)
            sigma_s -= np.einsum('ajbi,jb->ia', g[v, o, v, o], cis_sigma, optimize=True)

            return sigma

        self.matrix = matvec
        self.transition_method = 'cis'

    def spin_adapted_CIS(self, type='singlet'):
        #compute the spin-adapted CIS Hamiltonian

        #get mo basis 2-electron integrals and Fock
        g = mos.orbital_transform(self.scf, 'm', self.scf.get('i'))          
        f = mos.orbital_transform(self.scf, 'm', self.scf.get('f'))

        #orbital occupations and orbital slices
        nocc = sum(self.scf.mol.nele)//2  
        nvir = self.scf.mol.norb - nocc                                    
        o, v = slice(None, nocc), slice(nocc, None)                           

        ones = np.ones(self.scf.mol.norb)

        #compute the singlets
        if type == 'singlet':

            h =  np.einsum('ab,ij->iajb',np.diag(np.diag(f)[v]),np.diag(ones[o]))
            h -= np.einsum('ij,ab->iajb',np.diag(np.diag(f)[o]),np.diag(ones[v]))
            h += (2.0 *  g[o, v, o, v] - g[v, v, o, o].transpose(2,0,3,1))

            self.transition_method = 'cis'

        #compute the triplets
        if type == 'triplet':

            h =  np.einsum('ab,ij->iajb',np.diag(np.diag(f)[v]),np.diag(ones[o]))
            h -= np.einsum('ij,ab->iajb',np.diag(np.diag(f)[o]),np.diag(ones[v]))
            h -= g[v, v, o, o].transpose(2,0,3,1)

        #reshape and assign the class variable
        h = h.reshape(nocc*nvir, nocc*nvir)
        self.hamiltonian = h

        #Davidson methods
        self.davidson = True
        ds = mos.orbital_deltas(self.scf, 1, 'x')[0]

        def diagonal(self):
            #CIS diagonal for Davidson iterations

            #initialize to fock diagonal
            diagonal = -ds.ravel()

            cis_diagonal = diagonal[:(nocc*nvir)].reshape(nocc, nvir)
            if type == 'singlet':
                cis_diagonal += 2.0 * np.einsum('iaia->ia', g[o, v, o, v], optimize=True)
            cis_diagonal -= np.einsum('iaia->ia', g[v, v, o, o].transpose(2,0,3,1), optimize=True)

            return diagonal

        self.diagonal = diagonal

        def guess(self, roots, f=1):
            #initial vector to start Davidson

            diag = self.diagonal(self)

            #get largest absolute values on diagonal matrix as best guess
            args = np.argsort(np.absolute(diag))

            #we only have nocc*nvir roots available
            if roots > len(args):
                print('reducing requested roots - exceeded ', len(args))
                roots = len(args)

            guess_vectors = np.zeros((diag.size, roots * f))
            for root in range(roots * f):
                guess_vectors[args[root], root] = 1.0

            return np.array(guess_vectors)

        self.guess = guess

        def matvec(self, cis):
            #construct the self blocks of EE-self first order matrix dot product with arbitary vector (r)

            cis    = np.array(cis)
            sigma  = np.zeros_like(cis)

            cis_sigma = cis[:(nocc*nvir)].reshape(nocc, nvir)
            sigma_s  = sigma[:(nocc*nvir)].reshape(nocc, nvir)

            #singles sigmas
            sigma_s -= np.einsum('ia,ia->ia', ds, cis_sigma, optimize=True)
            if type == 'singlet':
                sigma_s += 2.0 * np.einsum('jbia,jb->ia', g[o, v, o, v], cis_sigma, optimize=True)
            sigma_s -= np.einsum('jaib,jb->ia', g[v, v, o, o].transpose(2,0,3,1), cis_sigma, optimize=True)

            return sigma

        self.matrix = matvec

    def RPA(self, type='TDA'):
        #Random Phase Approximation

        #get mo spin-basis 2-electron integrals and Fock
        g = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))          
        f = mos.orbital_transform(self.scf, 'm+s', self.scf.get('f'))

        #orbital occupations and orbital slices
        nocc = sum(self.scf.mol.nele)
        nvir = self.scf.mol.norb*2 - nocc                                    
        o, v = slice(None, nocc), slice(nocc, None)                           

        #compute the A and B ecitation and de-excitation matrices
        ones = np.ones(nocc + nvir)
        A =  np.einsum('ab,ij->iajb',np.diag(np.diag(f)[v]),np.diag(ones[o]))
        A -= np.einsum('ij,ab->iajb',np.diag(np.diag(f)[o]),np.diag(ones[v]))

        A = A.reshape(nocc*nvir, nocc*nvir)
        A += g[v, o, o, v].transpose(2,0,1,3).reshape(nocc*nvir, nocc*nvir)

        B =  g[v, v, o, o].transpose(2,0,3,1).reshape(nocc*nvir, nocc*nvir)

        if type == 'block':
            h = np.block([[ A,    B],
                          [-B,   -A]])
        if type == 'linear':
            h = np.einsum('pr,rq->pq', A+B, A-B, optimize=True)
            h = sp.linalg.sqrtm(h)

        if type == 'hermitian':
            sqrt_a_diff = sp.linalg.sqrtm(A-B)
            h = np.einsum('pr,rs,sq->pq', sqrt_a_diff, A+B, sqrt_a_diff, optimize=True)
            h = sp.linalg.sqrtm(h)  
            
        if type == 'TDA': 
            self.transition_method = 'cis'  
            h = A    

        if type == 'AB':
            return [A, B]

        if not 'h' in locals():
            h = np.zeros_like(A)

        self.hamiltonian = h

    def CIS_MP2(self, roots=10):
        #CIS with MP2 correction

        #orbital occupations and orbital slices
        nocc = sum(self.scf.mol.nele) 
        nvir = self.scf.mol.norb*2 - nocc                                    
        o, v, n = slice(None, nocc), slice(nocc, None), np.newaxis                          

        #get spin eri and orbital energies
        g  = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))
        eps = mos.orbital_transform(self.scf, 'm+s', self.scf.get('e'))

        #get CIS solution
        self.CIS()
        solve = solver(roots=roots, vectors=True)
        solve.direct(self.hamiltonian)

        #write corrections to class variable
        self.correction = []

        for root, value in enumerate(solve.values):

            #get amplitude vector and reshape
            b = solve.vectors[:, root]

            b = b.reshape(nocc, nvir)

            #shifted denominators
            dd_omega =  np.reciprocal(eps[o, n, n, n] + eps[n, o, n, n] 
                                    - eps[n, n, v, n] - eps[n, n, n, v] + value)

            dt_omega =  np.reciprocal(eps[o, n, n, n, n, n] + eps[n, o, n, n, n, n]
                                    + eps[n, n, o, n, n, n] - eps[n, n, n, v, n, n]
                                    - eps[n, n, n, n, v, n] - eps[n, n, n, n, n, v] + value)

            #u tensor
            U =  np.einsum('icab,jc->ijab', g[o,v,v,v], b, optimize=True)
            U -= np.einsum('jcab,ic->ijab', g[o,v,v,v], b, optimize=True)
            U += np.einsum('ijka,kb->ijab', g[o,o,o,v], b, optimize=True)
            U -= np.einsum('ijkb,ka->ijab', g[o,o,o,v], b, optimize=True)
            e_cis_mp2 = 0.25 * np.einsum('ijab,ijab->', U*U, dd_omega, optimize=True)

            U =  np.einsum('jkbc,ia->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('jkca,ib->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('jkab,ic->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('kibc,ja->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('kica,jb->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('kiab,jc->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('ijbc,ka->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('ijca,kb->ijkabc', g[o,o,v,v], b, optimize=True)
            U += np.einsum('ijab,kc->ijkabc', g[o,o,v,v], b, optimize=True)
            e_cis_mp2 += (1/36) * np.einsum('ijkabc,ijkabc->', U*U, dt_omega, optimize=True)

            self.correction.append([value, e_cis_mp2])

    def CIS_D(self, roots=10):
        #CIS(D) correction

        #orbital occupations and orbital slices
        nocc = sum(self.scf.mol.nele) 
        nvir = self.scf.mol.norb*2 - nocc                                    
        o, v, n = slice(None, nocc), slice(nocc, None), np.newaxis                          

        #get spin eri and orbital energies
        g   = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))
        eps = mos.orbital_transform(self.scf, 'm+s', self.scf.get('e'))

        #get CIS solution
        self.CIS()
        solve = solver(roots=roots, vectors=True)
        solve.direct(self.hamiltonian)

        #orbital denominators
        deltas = mos.orbital_deltas(self.scf, 2)
        eps_denominator = np.reciprocal(deltas[1]) 

        #write corrections to class variable
        self.correction = []

        for root, value in enumerate(solve.values):

            #get amplitude vector and reshape
            b = solve.vectors[:, root]

            b = b.reshape(nocc, nvir)

            #A tensor
            a = g[o, o, v, v] * eps_denominator

            #U tensor
            U =  np.einsum('icab,jc->ijab', g[o,v,v,v], b, optimize=True)
            U -= np.einsum('jcab,ic->ijab', g[o,v,v,v], b, optimize=True)
            U += np.einsum('ijka,kb->ijab', g[o,o,o,v], b, optimize=True)
            U -= np.einsum('ijkb,ka->ijab', g[o,o,o,v], b, optimize=True)

            #V tensor
            V =  0.5 * np.einsum('jkbc,ib,jkca->ia', g[o,o,v,v], b, a, optimize=True)
            V += 0.5 * np.einsum('jkbc,ja,ikcb->ia', g[o,o,v,v], b, a, optimize=True)
            V += np.einsum('jkbc,jb,ikac->ia', g[o,o,v,v], b, a, optimize=True)

            #double excitations - correlation term electrons not involved in excitation
            e_cisd_indirect  = np.einsum('ia,ia->', b, V, optimize=True)

            #shifted denominator
            dd_omega =  np.reciprocal(eps[o, n, n, n] + eps[n, o, n, n] 
                                    - eps[n, n, v, n] - eps[n, n, n, v] + value)

            e_cisd_direct = 0.25 * np.einsum('ijab,ijab->', U*U, dd_omega, optimize=True)

            #MP2 energy
            mp2_energy = 0.25 * np.einsum('ijab,ijab->', g[o, o, v, v], a, optimize=True)

            self.correction.append([value, e_cisd_indirect + e_cisd_direct + mp2_energy])

    def spatial_transition_properties(self, eig, roots):
        #get the transition dipole and oscillator for spin-adapted CIS

        #transform dipole in AO basis to MO basis - dipole gauge is chrge center
        dipoles = -np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.charge_center())) 
        mu = np.array([mos.orbital_transform(self.scf, 'm', dipoles[i]) for i in range(3)])

        #spatial orbital slices
        ndocc, nvir = self.scf.mol.nele[0], self.scf.mol.norb - self.scf.mol.nele[0]
        o, v = slice(None, ndocc), slice(ndocc, None)
 
        properties = []
        for root in range(roots):

            #compute the transition dipole 
            tdm = eig.vectors[:, root].reshape(ndocc, nvir)
            transition_dipole = np.einsum('ia,xia->x', tdm, mu[:, o, v], optimize=True) * np.sqrt(2)

            #oscillator strength
            oscillator = (2/3) * eig.values[root] * np.einsum('p,p->', transition_dipole, transition_dipole, optimize=True)
            
            if sum(abs(transition_dipole)) > 1e-8: properties.append([root, eig.values[root], transition_dipole, 
                                                                      oscillator, CI.transition(tdm, ndocc, nvir, spatial=True)])   

        return properties

    def transition(tdm, nocc, nvir, spatial=False):
        #generate the details of the transition

        levels = np.unravel_index(np.argmax(np.abs(tdm)), (nocc, nvir))
        value = int(np.abs(tdm[levels[0], levels[1]])*100)

        #correct for spin and virtual orbitals
        levels = [levels[0]//2, (levels[1] + nocc)//2] if not spatial else [levels[0], levels[1] + nocc]

        return str(levels[0]).ljust(2) + '-> ' + str(levels[1]).ljust(2) + ' (' + str(value) + '%)'

    def transition_density(self, eig, root):
        #reshape CIS coefficients for transition density

        if hasattr(eig, 'vectors'):
            return eig.vectors[:, root]
        else:
            exit('no eigensolution')

    def transition_properties(self, eig, roots=5, type='electric dipole'):
        #compute the transition property requested

        if roots == -1: roots = len(eig.values)

        if eig.vectors.shape[0] == self.scf.mol.nele[0]*(self.scf.mol.norb - self.scf.mol.nele[0]):
            return self.spatial_transition_properties(eig, roots)

        nocc = sum(self.scf.mol.nele)
        nvir = self.scf.mol.norb * 2 - nocc

        if self.transition_method == '': return []
        if self.transition_method == 'fci':
            eig.values -= self.scf.reference[0] 

            #fermi vacuum translation
            idx = list(zip(*(iter(list(range(nocc*nvir))[::-1]),) * nvir))
            idx = np.array(sum([list(i)[::-1] for i in idx], []))
            sign = np.array([1*((-1)**j)  for j in range(nocc) for i in range(nvir)])

        #orbital slices
        o = slice(None, sum(self.scf.mol.nele))
        v = slice(sum(self.scf.mol.nele), None)
        
        properties = []
        if type == 'electric dipole':

            #transform dipole in AO basis to MO spin basis - dipole gauge is charge center
            dipoles = -np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.charge_center())) 
            mu = np.array([mos.orbital_transform(self.scf, 'm+s', dipoles[i]) for i in range(3)])

            for root in range(roots):

                #compute the transition dipole 
                tdm = self.transition_density(eig, root)
                if self.transition_method == 'fci':
                    tdm = sign * tdm[idx]
                tdm = tdm.reshape(nocc, nvir)

                transition_dipole = np.einsum('ia,xia->x', tdm, mu[:, o, v], optimize=True)

                #oscillator strength
                oscillator = (2/3) * eig.values[root] * np.einsum('p,p->', transition_dipole, transition_dipole, optimize=True)
                
                if sum(abs(transition_dipole)) > 1e-8: properties.append([root, eig.values[root], transition_dipole,
                                                                          oscillator, CI.transition(tdm, nocc, nvir)])

            return properties

