from __future__ import division

import numpy as np

import int.mo_spin as mos
from phf.mpt import MP
from int.aello import aello

'''
ADC - Development and implementation of theoretical methods for the description of
      electronically core-excited states by J Wenzel
      - [https://archiv.ub.uni-heidelberg.de/volltextserver/20514/1/Jan_Wenzel_Thesis.pdf]
    - Develpment and application of hermitian methods for molecular properties and
      excited electronic states by M Hodecker -[https://core.ac.uk/download/pdf/322693292.pdf]
'''

#*******************************************
#* Algebraic Diagrammatic Construction (2) *
#*******************************************

class ADC(object):
    #Algebraic Diagrammatic Construction at order 2

    def __init__(self, scf):

        self.scf = scf
        self.has_reduced, self.method, self.cache = False, '', {}

        self.shared()
        self.set_options()

    def execute(self, code, eigensolver, roots=3, solution='direct'):
        #dispatcher for execution

        if not code in ['ee', 'ip', 'ea']:
            print('ADC(2) supports EE, IP and EA - not ', code)
            return

        if code == 'ee': self._ee(eigensolver, roots, solution)
        if code == 'ip': self._ip(eigensolver, roots, solution)
        if code == 'ea': self._ea(eigensolver, roots, solution)

    def analyse(self):
        #dispatcher for eigenvector analysis

        if self.method == 'ee': self.ee_analyse()
        if self.method == 'ip': self._analyse()
        if self.method == 'ea': self._analyse()

    def set_options(self, threshold=0.1, leading=None):
        #set the conditions for printing excitations

        self.options = {}
        self.options['THRESHOLD'] = threshold
        self.options['LEADING']   = leading
        if not leading is None: self.options['THRESHOLD'] = 0.02

    def adc_degeneracies(self):
        #return tuples of root value and its multiplicity

        from phf.cit import CI
        return CI.degeneracies(None, self.cache['values'])

    def shared(self):
        #define and compute common properties

        #molecular orbital energy differences and 2-electron spin mo integrals
        self.deltas = mos.orbital_deltas(self.scf, 2)
        self.gs     = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))
        self.es     = mos.orbital_transform(self.scf, 's', self.scf.get('e'))

        #occupations
        self.nocc = sum(self.scf.mol.nele)
        nmo  = self.scf.mol.norb * 2
        self.nvir = nmo - self.nocc

        n, o, v = np.newaxis, slice(None, self.nocc), slice(self.nocc, None)

        #amplitude initialisation
        self.td = self.gs[o, o, v, v] * np.reciprocal(self.deltas[1])

        def guess(self, roots, f=2):
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

    def condense_matrix(self):
        #get the dense version of an ADC matrix from vectors

        if callable(self.matrix):
            matvec = self.matrix

            #get tensor dimensions and define
            diagonal_size = self.diagonal(self).size
            vector = np.zeros(diagonal_size)
            matrix = np.zeros((diagonal_size, diagonal_size))

            vector[0] = 1.0
            for p in range(diagonal_size):
                matrix[:, p] = matvec(self, vector)
                vector = np.roll(vector, 1)

        return matrix

    def matrix_partition_information(self):
        #give details of size of matrix blocks

        block = {}
        if self.method == 'ee':

            block['label'] = ('1h-1p', '2h-2p')
            block['size'] = (self.nocc * self.nvir,
                             self.nocc * self.nocc * self.nvir * self.nvir)
            block['shape'] = ((self.nocc, self.nvir),
                              (self.nocc, self.nocc, self.nvir, self.nvir))
            block['orbital'] = ([1, 1+self.nocc//2],[1, 1, 1+self.nocc//2, 1+self.nocc//2])

        if self.method == 'ip':

            block['label'] = ('1h', '2h-1p')
            block['size'] = (self.nocc,
                             self.nocc * self.nocc * self.nvir)
            block['shape'] = ((self.nocc,),
                              (self.nocc, self.nocc, self.nvir))
            block['orbital'] = ([1], [1, 1, 1+self.nocc//2])

        if self.method == 'ea':

            block['label'] = ('1p', '1h-2p')
            block['size'] = (self.nvir,
                                 self.nocc * self.nvir * self.nvir)
            block['shape'] = ((self.nvir,),
                                  (self.nocc, self.nvir, self.nvir))
            block['orbital'] = ([1+self.nocc//2], [1, 1+self.nocc//2, 1+self.nocc//2])

        self.block = block

    def norms(self, root):
        #compute the norms

        #singles
        v = self.cache['vectors'][:self.block['size'][0], root]
        norm_single = np.einsum('i,i->', v, v, optimize=True)

        #doubles
        v = self.cache['vectors'][self.block['size'][0]:, root]
        norm_double = np.einsum('i,i->', v, v, optimize=True)

        adc_norm = [self.block['label'], float(norm_single), float(norm_double)]

        return adc_norm

    def transition_density(self, root):
        #compute the transition density for eigenvector 'root'

        if self.method != 'ee':
            print('transition properties only available for EE')
            return

        #eigenvectors
        us = self.cache['vectors'][:self.block['size'][0], root].reshape(self.block['shape'][0])
        ud = self.cache['vectors'][self.block['size'][0]:, root].reshape(self.block['shape'][1])

        #orbital slices
        o, v = slice(None, self.nocc), slice(self.nocc, None)

        t = np.einsum('imae,mbje->ijab', self.td, self.gs[o, v, o, v], optimize=True)
        tD  = t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)
        tD -= 0.5 * (np.einsum('ijef,abef->ijab', self.td, self.gs[v, v, v, v]) +
                     np.einsum('mnab,mnij->ijab', self.td, self.gs[o, o, o, o], optimize=True))
        tD *= np.reciprocal(self.deltas[1])

        #define transition density matrix
        tdm = np.zeros((self.nocc+self.nvir, self.nocc+self.nvir))

        #0th order contribution
        tdm[v, o] += us.transpose(1, 0)
        #1st order contribution
        tdm[o, v] += np.einsum('ijab,jb->ia', self.td, us, optimize=True)

        #get mp2 relaxed density matrix
        mp = MP(self.scf, method='MP2rdm1', parameter='relaxed')
        mpdm = mp.mp2dm

        #2nd order contributions
        tdm[o, o] -= np.einsum('ia,ja->ij', mpdm[o, v], us, optimize=True)
        tdm[o, o] += np.einsum('imab,jmab->ij', ud, self.td, optimize=True)

        tdm[v, v] += np.einsum('ia,ib->ab', us, mpdm[o,v], optimize=True)
        tdm[v, v] -= np.einsum('ijae,ijbe->ab', ud, self.td, optimize=True)

        tdm[o, v] -= np.einsum('ijab,jb->ia', tD, us, optimize=True)

        tdm[v, o] += 0.5 * (np.einsum('ijab,jmbe,me->ai', self.td, self.td, us, optimize=True)
                           -np.einsum('ab,ib->ai', mpdm[v, v], us, optimize=True)
                           +np.einsum('ja,ij->ai', us, mpdm[o, o] - np.eye(self.nocc)[o,o], optimize=True))

        return tdm

    def transition_properties(self, root):
        #compute the transition dipole and electric oscillator strength

        #get transition density
        tdm = self.transition_density(root)

        #get dipole components from aello and -> mo spin basis
        dipoles = np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.charge_center()))
        mu_mo = np.array([mos.orbital_transform(self.scf, 'm+s', dipoles[i]) for i in range(3)])

        #get transition dipole moment and oscillator strength
        tmu = np.einsum('ia,xia->x', tdm, mu_mo, optimize=True)
        os = (2/3) * self.cache['values'][root] * np.einsum('x,x->', tmu, tmu, optimize=True)

        return tmu, os

    def _ee(self, eigensolver, roots=3, solution='direct'):
        #execute an electron excitation

        self.method, self.cache = 'ee', {}
        self.roots = roots

        #shortforms
        deltas, gs, td = self.deltas, self.gs, self.td
        n, o, v = np.newaxis, slice(None, self.nocc), slice(self.nocc, None)
        nocc, nvir = self.nocc, self.nvir

        def diagonal(self):
            #ADC-EE matrix diagonal preconditioner for Davidson

            #initialize to fock diagonal
            diagonal = -np.concatenate([deltas[0].ravel(), deltas[1].ravel()])

            adc_diagonal = diagonal[:(nocc*nvir)].reshape(nocc, nvir)

            adc_diagonal -= np.einsum('aiai->ia', gs[v, o, v, o], optimize=True)

            adc_diagonal += 0.5  * np.einsum('aeim,imae->ia', gs[v, v, o, o], td, optimize=True)
            adc_diagonal += 0.5  * np.einsum('imae,imae->ia', gs[o, o, v, v], td, optimize=True)
            adc_diagonal -= 0.25 * np.einsum('efim,imef->i',  gs[v, v, o, o], td, optimize=True)[:, n]
            adc_diagonal -= 0.25 * np.einsum('imef,imef->i',  gs[o, o, v, v], td, optimize=True)[:, n]
            adc_diagonal -= 0.25 * np.einsum('aemn,mnae->a',  gs[v, v, o, o], td, optimize=True)[n, :]
            adc_diagonal -= 0.25 * np.einsum('mnae,mnae->a',  gs[o, o, v, v], td, optimize=True)[n, :]

            return diagonal

        self.diagonal = diagonal

        def matvec(self, adc):
            #construct the ADC blocks of EE-ADC second order matrix dot product with arbitary vector

            adc = np.array(adc)
            r   = np.zeros_like(adc)

            R, ADC = [None, None], [None, None]
            ADC[0] = adc[:(nocc*nvir)].reshape(nocc, nvir)
            R[0]   = r[:(nocc*nvir)].reshape(nocc, nvir)

            #ph-ph block
            R[0] -= np.einsum('ia,ia->ia', deltas[0], ADC[0], optimize=True)

            R[0] -= np.einsum('ajbi,jb->ia', gs[v, o, v, o], ADC[0], optimize=True)

            R[0] += 0.5 * np.einsum('aeim,jmbe,jb->ia', gs[v, v, o, o], td, ADC[0], optimize=True)
            R[0] += 0.5 * np.einsum('jmbe,imae,jb->ia', gs[o, o, v, v], td, ADC[0], optimize=True)

            t     = -np.einsum('efim,jmef->ij', gs[v, v, o, o], td, optimize=True)
            t    += -np.einsum('jmef,imef->ij', gs[o, o, v, v], td, optimize=True)
            R[0] += 0.25 * np.einsum('ij,ja->ia', t, ADC[0], optimize=True)

            t     = -np.einsum('aemn,mnbe->ab', gs[v, v, o, o], td, optimize=True)
            t    += -np.einsum('mnbe,mnae->ab', gs[o, o, v, v], td, optimize=True)
            R[0] += 0.25 * np.einsum('ab,ib->ia', t, ADC[0], optimize=True)

            ADC[1] = adc[(nocc*nvir):].reshape(nocc, nocc, nvir, nvir)
            R[1]   = r[(nocc*nvir):].reshape(nocc, nocc, nvir, nvir)

            #ph - pphh block
            R[0] += 0.5 * np.einsum('mnie,mnae->ia', gs[o, o, o, v], ADC[1], optimize=True)
            R[0] -= 0.5 * np.einsum('mnie,mnea->ia', gs[o, o, o, v], ADC[1], optimize=True)
            R[0] -= 0.5 * np.einsum('amef,imef->ia', gs[v, o, v, v], ADC[1], optimize=True)
            R[0] += 0.5 * np.einsum('amef,mief->ia', gs[v, o, v, v], ADC[1], optimize=True)

            #pphh - ph block
            R[1] += 0.5 * np.einsum('mbij,ma->ijab', gs[o, v, o, o], ADC[0], optimize=True)
            R[1] -= 0.5 * np.einsum('maij,mb->ijab', gs[o, v, o, o], ADC[0], optimize=True)
            R[1] -= 0.5 * np.einsum('abej,ie->ijab', gs[v, v, v, o], ADC[0], optimize=True)
            R[1] += 0.5 * np.einsum('abei,je->ijab', gs[v, v, v, o], ADC[0], optimize=True)

            #pphh - pphh block
            R[1] -= np.einsum('ijab,ijab->ijab', deltas[1], ADC[1], optimize=True)

            return r

        self.matrix = matvec

        #use Davidson iterative solver or dense matrix
        eigensolver.roots = self.roots
        if solution == 'iterative': eigensolver.iterative(self)
        if solution == 'direct':
            h = self.condense_matrix()
            eigensolver.direct(h)

        self.cache['values'], self.cache['vectors'] = eigensolver.values, eigensolver.vectors

        return

    def ee_analyse(self):
        #do an analysis of EE eigenvectors

        if self.method != 'ee':
            print('Type ', self.method, ' and method \'EE_analyse\' are incompatible')
            return

        #get matrix partioning and size information
        self.matrix_partition_information()

        def components(u):
            #return the spin-components of eigenvector u
            if u.ndim == 2:
                return [u[::2,::2].ravel(),  u[1::2,1::2].ravel(),
                        u[::2,1::2].ravel(), u[1::2,::2].ravel()]
            if u.ndim == 4:
                return [u[::2,::2,::2,::2].ravel(),   u[1::2,1::2,1::2,1::2].ravel(),
                        u[::2,1::2,::2,1::2].ravel(), u[1::2,::2,1::2,::2].ravel()]

        #spin symbols
        symbol = ['\u2963', '\u296E']

        #cache dictionary list properties
        self.cache['excitations'], self.cache['norms'], self.cache['transition properties'] = [], [], []

        for root in range(self.roots):

            #add norms to cache
            self.cache['norms'].append(self.norms(root))

            excitations = []
            #1h-1p block excitations
            u = (self.cache['vectors'][:self.block['size'][0], root]**2.0).reshape(self.block['shape'][0])

            x = np.sqrt(sum(components(u))) / pow(2, 0.5)

            #find largest element
            idx = np.argsort(x)[::-1]
            x = x[idx]

            spatial = [i//2 for i in self.block['shape'][0]]
            dominant_element_count = np.count_nonzero(x > self.options['THRESHOLD'])

            if dominant_element_count:
                ix = np.vstack(np.unravel_index(idx[:dominant_element_count], tuple(spatial))).transpose().tolist()
                x = x[:dominant_element_count]

                #get spin type a->a or a->b
                spin_excitation = (np.vstack(np.unravel_index(idx[:dominant_element_count],
                                   self.block['shape'][0])).transpose().tolist())[0]
                spin = 0 if (sum(spin_excitation) % 2) == 0 else 1

                #add state information - corrected for orbital indexing from 1
                if ix != []:
                    bias = self.block['orbital'][0]

                    if not self.options['LEADING'] is None:
                        effective_leading = min(dominant_element_count, self.LEADING)
                        x, ix = x[:effective_leading], ix[:effective_leading]

                    ix_orbitals = []
                    for i in ix:
                        ix_orbitals.append([j+bias[n] for n, j in enumerate(i)])

                    excitations.append([self.block['label'][0], list(x), ix_orbitals, symbol[spin]])

            #2p-2h excitations
            u = (self.cache['vectors'][self.block['size'][0]:, root]**2.0).reshape(self.block['shape'][1])

            same_spin, cross_spin = (np.sqrt(2*sum(components(u)[:2])),
                                     np.sqrt(2*sum(components(u)[2:])))

            spatial = [i//2 for i in self.block['shape'][1]]

            for spin, x in enumerate([same_spin, cross_spin]):

                #find largest element
                x *= 0.5
                idx = np.argsort(x)[::-1]
                x = x[idx]

                dominant_element_count = np.count_nonzero(x > self.options['THRESHOLD'])

                if dominant_element_count:
                    ix = np.vstack(np.unravel_index(idx[:dominant_element_count], tuple(spatial))).transpose().tolist()
                    x = [min(i, 1.0) for i in x[:dominant_element_count]]

                    #add state information
                    if ix != []:
                        bias = self.block['orbital'][1]

                        if not self.options['LEADING'] is None:
                            effective_leading = min(dominant_element_count, self.options['LEADING'])
                            x, ix = x[:effective_leading], ix[:effective_leading]

                        ix_orbitals = []
                        for i in ix:
                            ix_orbitals.append([j+bias[n] for n, j in enumerate(i)])

                        excitations.append([self.block['label'][1], list(x), ix_orbitals, symbol[spin]])

            self.cache['excitations'].append(excitations)

            #get the transition dipole and oscillator strengths
            dipole, oscillator = self.transition_properties(root)
            self.cache['transition properties'].append([dipole, oscillator])

    def amplitudes(self):
        #constructs the ADC(2) amplitudes

        #orbital slices
        o, v = slice(None, self.nocc), slice(self.nocc, None)

        td = self.gs[o, o, v, v] * np.reciprocal(self.deltas[1])
        ts = -0.5 * (np.einsum('jkab,jkib->ia', td, self.gs[o, o, o, v], optimize=True) +
                     np.einsum('ijbc,jabc->ia', td, self.gs[o, v, v, v], optimize=True)) * np.reciprocal(self.deltas[0])

        return ts, td

    def effective_transition_amplitudes(self):
        #compute the effective transition amplitudes (f)

        #get amplitudes
        ts, td = self.amplitudes()

        if self.method == 'ip':
            dimension = (sum(list(self.block['size'])), self.nocc + self.nvir)
            f = np.zeros(dimension)

            #orbital slices
            o, v = slice(None, self.nocc), slice(self.nocc, None)

            f[o, o] += np.eye(self.nocc)
            f[o, o] += -0.25 * np.einsum('ikab,jkab->ij', td, td, optimize=True)

            f[o, v] += ts

            f[v, v] += td.reshape(self.block['size'][1], self.nvir) * np.sqrt(0.5)

        if self.method == 'ea':
            dimension = (self.nocc + self.nvir, sum(list(self.block['size'])))
            f = np.zeros(dimension)

            #orbital slices
            v, o = slice(None, self.nvir), slice(self.nvir, None)

            f[v, v] += np.eye(self.nvir)
            f[v, v] += -0.25 * np.einsum('ijbc,ijac->ab', td, td, optimize=True)

            f[o, v] -= ts

            f[o, o] -= td.reshape(self.nocc, self.block['size'][1]) * np.sqrt(0.5)

        return f

    def _ip(self, eigensolver, roots=3, solution='direct'):
        #execute an ionisation potential

        self.method, self.cache = 'ip', {}
        self.roots = roots

        #shortforms
        deltas, gs, es, td = self.deltas, self.gs, self.es, self.td
        n, o, v = np.newaxis, slice(None, self.nocc), slice(self.nocc, None)
        nocc, nvir = self.nocc, self.nvir

        #reduce o-o subspace
        i, j = np.tril_indices(nocc, -1)
        nsub = len(i)

        #self-energy static block
        f = np.diag(es[o])
        f += 0.25 * np.einsum('imab,jmab->ij', td, gs[o, o, v, v], optimize=True)
        f += 0.25 * np.einsum('jmab,imab->ij', td, gs[o, o, v, v], optimize=True)

        #diagonal contribution
        k = es[o, n, n] + deltas[0][n]
        k = k[i, j, :]

        def matvec(self, adc):

            adc = np.array(adc)
            r = np.zeros_like(adc)

            R   = [r[:nocc],     r[nocc:].reshape(  len(i), nvir)]
            ADC = [adc[:nocc], adc[nocc:].reshape(  len(i), nvir)]

            R[0] += np.dot(f, ADC[0])
            R[0] += np.einsum('iak,ia->k', gs[:nocc, :nocc, nocc:, :nocc][i,j,:,:].reshape(len(i), nvir, nocc), ADC[1])

            R[1] += np.einsum('iak,k->ia', gs[:nocc, :nocc, nocc:, :nocc][i,j,:,:].reshape(len(i), nvir, nocc), ADC[0])
            R[1] += np.einsum('ia,ia->ia', k.reshape(-1, nvir), ADC[1])

            return r

        self.matrix = matvec

        def diagonal(self):

            return np.concatenate([np.diag(f), k.ravel()])

        self.diagonal = diagonal

        #use Davidson iterative solver or dense matrix
        eigensolver.roots = self.roots
        if solution == 'iterative': eigensolver.iterative(self)
        if solution == 'direct':
            eigensolver.roots = -eigensolver.roots
            h = self.condense_matrix()
            eigensolver.direct(h)

        self.cache['values'], self.cache['vectors'] = np.abs(eigensolver.values), eigensolver.vectors

        return

    def _ea(self, eigensolver, roots=3, solution='direct'):
        #execute an ionisation potential

        self.method, self.cache = 'ea', {}
        self.roots = roots

        #shortforms
        deltas, gs, es, td = self.deltas, self.gs, self.es, self.td
        n, o, v = np.newaxis, slice(None, self.nocc), slice(self.nocc, None)
        nocc, nvir = self.nocc, self.nvir

        #reduce v-v subspace
        a, b = np.tril_indices(nvir, -1)
        nsub = len(a)

        #self-energy static block
        f = np.diag(es[v])
        f -= 0.25 * np.einsum('ijae,ijbe->ab', td, gs[o, o, v, v], optimize=True)
        f -= 0.25 * np.einsum('ijbe,ijae->ab', td, gs[o, o, v, v], optimize=True)

        #diagonal contribution
        k = deltas[0][:, :, n] - es[n, n, v]
        k = -k[:, a, b]

        def matvec(self, adc):

            adc = np.array(adc)
            r = np.zeros_like(adc)

            R   = [r[:nvir],     r[nvir:].reshape(nocc, nsub)]
            ADC = [adc[:nvir], adc[nvir:].reshape(nocc, nsub)]

            R[0] += np.dot(f, ADC[0])
            R[0] -= np.einsum('cia,ia->c', gs[nocc:, :nocc, nocc:, nocc:][:,:,a,b].reshape(nvir, nocc, nsub), ADC[1])

            R[1] -= np.einsum('cia,c->ia', gs[nocc:, :nocc, nocc:, nocc:][:,:,a,b].reshape(nvir, nocc, nsub), ADC[0])
            R[1] += np.einsum('ia,ia->ia', k.reshape(nocc, -1), ADC[1])

            return r

        self.matrix = matvec

        def diagonal(self):

            return np.concatenate([np.diag(f), k.ravel()])

        self.diagonal = diagonal

        #use Davidson iterative solver or dense matrix
        eigensolver.roots = self.roots
        if solution == 'iterative': eigensolver.iterative(self)
        if solution == 'direct':
            h = self.condense_matrix()
            eigensolver.direct(h)

        self.cache['values'], self.cache['vectors'] = eigensolver.values, eigensolver.vectors

        return

    def _analyse(self):
        #do an analysis of IP or EA eigenvectors

        if not self.method in ['ip', 'ea']:
            print('Type ', self.method, ' and method \'EA_IP__analyse\' are incompatible')
            return

        #spin symbols
        symbol = ['\u2963', '\u296E']

        #get matrix partioning and size information
        self.matrix_partition_information()

        def components(u):
            #return the spin-components of eigenvector u
            if u.ndim == 1:
                return [u[::2].ravel(), u[1::2].ravel()]
            if u.ndim == 3:
                return [u[::2,::2,::2].ravel(),  u[1::2,1::2,1::2].ravel(),
                        u[::2,1::2,::2].ravel(), u[1::2,::2,1::2].ravel()]

        #get space reduction indices
        reduction = np.tril_indices(self.nvir, k=-1) if self.method == 'ea' else np.tril_indices(self.nocc, k=-1)

        #effective transition amplitudes
        transition_amplitudes = self.effective_transition_amplitudes()

        #cache dictionary lists
        self.cache['norms'], self.cache['excitations'], self.cache['spectral properties'] = [], [], []

        for root in range(self.roots):

            #add norms to cache
            self.cache['norms'].append(self.norms(root))

            excitations = []

            #1p block excitations
            u = (self.cache['vectors'][:self.block['size'][0], root]**2.0).reshape(self.block['shape'][0])
            x = np.sqrt(sum(components(u)))

            #find largest element
            idx = np.argsort(x)[::-1]
            x = x[idx]

            spatial = [i//2 for i in self.block['shape'][0]]
            dominant_element_count = np.count_nonzero(x > self.options['THRESHOLD'])

            if dominant_element_count:
                ix = np.vstack(np.unravel_index(idx[:dominant_element_count], tuple(spatial))).transpose().tolist()
                x = x[:dominant_element_count]

                #add state information - corrected for orbital indexing from 1
                if ix != []:
                    bias = self.block['orbital'][0]

                    if not self.options['LEADING'] is None:
                        effective_leading = min(dominant_element_count, self.LEADING)
                        x, ix = x[:effective_leading], ix[:effective_leading]

                    ix_orbitals = []
                    for i in ix:
                        ix_orbitals.append([j+bias[n] for n, j in enumerate(i)])

                    excitations.append([self.block['label'][0], list(x), ix_orbitals, ''])

            #1p-2h or 2p-1h excitations
            shape_ = (self.nocc, len(reduction[0])) if self.method == 'ea' else (len(reduction[0]), self.nvir)
            x = (self.cache['vectors'][self.block['size'][0]:, root]**2.0).reshape(shape_)

            #expand space
            u = np.array([])
            shape_ = self.block['shape'][1][1:] if self.method == 'ea' else self.block['shape'][1][:2]
            range_ = self.nocc if self.method == 'ea' else self.nvir
            for vdim in range(range_):
                subspace = np.zeros(shape_)
                subspace[reduction[0], reduction[1]] = x[vdim, :] if self.method == 'ea' else x[:, vdim]
                subspace += subspace.transpose()

                u = np.vstack([u, subspace.ravel()]) if u.size else subspace.ravel()
            u = u.reshape(self.block['shape'][1]) if self.method == 'ea' else u.transpose().reshape(self.block['shape'][1])

            same_spin, cross_spin = (np.sqrt(sum(components(u)[:2])),
                                     np.sqrt(sum(components(u)[2:])))

            spatial = [i//2 for i in self.block['shape'][1]]

            for spin, x in enumerate([same_spin, cross_spin]):

                #find largest element
                idx = np.argsort(x)[::-1]
                x = x[idx]

                dominant_element_count = np.count_nonzero(x > self.options['THRESHOLD'])

                if dominant_element_count:
                    ix = np.vstack(np.unravel_index(idx[:dominant_element_count], tuple(spatial))).transpose().tolist()
                    x = [min(i, 1.0) for i in x[:dominant_element_count]]

                    #add state information
                    if ix != []:
                        bias = self.block['orbital'][1]

                        if not self.options['LEADING'] is None:
                            effective_leading = min(dominant_element_count, self.options['LEADING'])
                            x, ix = x[:effective_leading], ix[:effective_leading]

                        ix_orbitals = []
                        for i in ix:
                            ix_orbitals.append([j+bias[n] for n, j in enumerate(i)])

                        excitations.append([self.block['label'][1], list(x), ix_orbitals, symbol[spin]])

            self.cache['excitations'].append(excitations)

            #compute spectroscopic factors
            expanded_vector = np.concatenate((self.cache['vectors'][:self.block['size'][0], root],
                                             u.ravel()))
            eta = self.effective_transition_amplitudes()
            string_ = 'px,x->p' if self.method == 'ea' else 'xp,x->p'
            x = np.einsum(string_, eta, expanded_vector, optimize=True)

            #spectroscopic factor
            x = x * x
            contributions = [x[i] + x[i+1] for i in range(0, len(x), 2)]

            bias = self.block['orbital'][0][0] if self.method == 'ea' else 0
            self.cache['spectral properties'].append([sum(contributions), np.argmax(contributions) + bias,
                                                      contributions[np.argmax(contributions)]])

