from __future__ import division
import numpy as np

from phf.eig import solver
from phf.cit import CI
from phf.cct import CC

'''
1. Chapter 13 - MANY-BODY METHODS IN CHEMISTRY AND PHYSICS MBPT and Coupled-Cluster Theory Shavitt & Bartlett
2. The equation of motion coupled-clustermethod. A systematic biorthogonal approach to molecular excitation energies,
   transition probabilities, and excited state properties - J. Chem. Phys. 98, 7029 (1993)
3. Simplified methods for equation-of-motion coupled-cluster excited state calculations - Steven R. Gwaltney, Marcel Nooijen, Rodney J. Bartlet
'''
class EOM(object):
    #Equation-of-motion Coupled Cluster Singles and Doubles

    def __init__(self, scf):

        #instance of RHF class
        self.scf = scf
        self.method, self.cache = '', {}

        #define common properties
        self.shared()
        self.set_options()

        #get one and two body Hamiltonian elements
        self.h = EOM.hbar(self.cc)

    def execute(self, code, eigensolver, left=False, roots=3, solution='iterative'):
        #dispatcher for execution

        code = code.lower()
        if not code in ['ee', 'ip', 'ea']:
            print('EOM-CCSD supports EE, IP and EA not ', code)
            return

        if code == 'ee': self._ee(eigensolver, roots, solution)
        if code == 'ip': self._ip(eigensolver, left, roots, solution)
        if code == 'ea': self._ea(eigensolver, left, roots, solution)

    def set_options(self, threshold=0.1, leading=None):
        #set the conditions for printing excitations

        self.options = {}
        self.options['THRESHOLD'] = threshold
        self.options['LEADING']   = leading
        if not leading is None: self.options['THRESHOLD'] = 0.02

    def eom_degeneracies(self):
        #return tuples of root value and its multiplicity

        return CI.degeneracies(None, self.cache['values'])

    def shared(self):
        #define and compute common properties

        #get instance of coupled-cluster class and run a CCSD calculation
        cc = CC(self.scf)
        cc.method('ccsd')
        self.cc = cc

        #molecular orbital energy differences and 2-electron spin mo integrals
        self.deltas = cc.deltas
        self.gs     = cc.gs

        #occupations
        self.nocc = sum(self.scf.mol.nele)
        nmo  = self.scf.mol.norb * 2
        self.nvir = nmo - self.nocc

        #amplitude initialisation
        self.td = cc.td

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

            i, j = np.tril_indices(self.nocc, k=-1)
            a, b = np.tril_indices(self.nvir, k=-1)

            block['label'] = ('1h-1p', '2h-2p')
            block['size'] = (self.nocc * self.nvir,
                             len(i) * len(a))
            block['shape'] = ((self.nocc, self.nvir),
                              (self.nocc, self.nocc, self.nvir, self.nvir))
            block['orbital'] = ([1, 1+self.nocc//2],[1, 1, 1+self.nocc//2, 1+self.nocc//2])

            def reduce(hp, hhpp):

                u = (hhpp[i, j, :, :].reshape(len(i), self.nvir, self.nvir)
                         [:, a, b].reshape(len(i), len(a)))

                return np.hstack((hp.ravel(), u.ravel()))

            def expand(u):
                hp = u[:block['size'][0]].reshape(block['shape'][0])

                intermediate = np.zeros((self.nocc, self.nocc, len(a)))
                intermediate[i,j,:] =  u[block['size'][0]:].reshape(len(i), -1)
                intermediate[j,i,:] = -u[block['size'][0]:].reshape(len(i), -1)

                hhpp = np.zeros(block['shape'][1])
                hhpp[:,:,a,b] =  intermediate.reshape(self.nocc, self.nocc, len(a))
                hhpp[:,:,b,a] = -intermediate.reshape(self.nocc, self.nocc, len(a))

                return hp, hhpp

        if self.method == 'ip':

            i, j = np.tril_indices(self.nocc, k=-1)

            block['label'] = ('1h', '2h-1p')
            block['size'] = (self.nocc,
                             len(i) * self.nvir)
            block['shape'] = ((self.nocc,),
                              (self.nocc, self.nocc, self.nvir))
            block['orbital'] = ([1], [1, 1, 1+self.nocc//2])

            def reduce(h, hp):
                return np.hstack((h, hp[i, j, :].ravel()))

            def expand(u):
                h = u[:self.nocc]
                hp = np.zeros(block['shape'][1])

                hp[i, j, :] =  u[self.nocc:].reshape(-1, self.nvir)
                hp[j, i, :] = -u[self.nocc:].reshape(-1, self.nvir)

                return h, hp

        if self.method == 'ea':

            a, b = np.tril_indices(self.nvir, k=-1)

            block['label'] = ('1p', '1h-2p')
            block['size'] = (self.nvir,
                             self.nocc * len(a))
            block['shape'] = ((self.nvir,),
                                  (self.nocc, self.nvir, self.nvir))
            block['orbital'] = ([1+self.nocc//2], [1, 1+self.nocc//2, 1+self.nocc//2])

            def reduce(p, hp):
                return np.hstack((p, hp[:, a, b].ravel()))

            def expand(u):
                p = u[:self.nvir]
                hp = np.zeros(block['shape'][1])

                hp[:, a, b] =  u[self.nvir:].reshape(self.nocc, -1)
                hp[:, b, a] = -u[self.nvir:].reshape(self.nocc, -1)

                return p, hp

        self.reduce = reduce
        self.expand = expand

        self.block = block

    def hbar(cc):
        #components of similarity-transformed CCSD Hamiltonian

        o, v = cc.o, cc.v
        h = (np.zeros_like(cc.fs), np.zeros_like(cc.gs))

        #1-body components
        h[0][o, v] = cc.intermediates('ov', tilde=False)
        h[0][o, o] = cc.intermediates('oo', tilde=False) + cc.fs[o, o]
        h[0][v, v] = cc.intermediates('vv', tilde=False) + cc.fs[v, v]

        #2-body components
        h[1][v, o, v, v] = cc.intermediates('vovv', tilde=False)
        h[1][o, o, o, v] = cc.intermediates('ooov', tilde=False)
        h[1][v, v, v, v] = cc.intermediates('vvvv', tilde=False)
        h[1][o, o, o, o] = cc.intermediates('oooo', tilde=False)
        h[1][v, o, o, v] = cc.intermediates('ovvo', tilde=False).transpose(1,0,3,2)
        h[1][v, o, o, o] = -cc.intermediates('ovoo', tilde=False).transpose(1,0,2,3)
        h[1][v, v, o, v] = -cc.intermediates('vvvo', tilde=False).transpose(0,1,3,2)
        h[1][o, o, v, v] = cc.gs[o, o, v, v]

        return h

    def _ee(self, eigensolver, roots=3, solution='iterative'):

        self.method, self.cache = 'ee', {}
        self.roots  = roots

        #shortform globals
        o, v, nocc, nvir = self.cc.o, self.cc.v, self.nocc, self.nvir
        td, h, deltas = self.td, self.h, self.deltas

        self.matrix_partition_information()

        def guess(self, roots):
            #use lowest state of CIS level as inital guess vectors

            ci = CI(self.scf)
            ci.CIS()

            eigensolver.direct(ci.hamiltonian)

            #orthonormalize the initialguess  space
            guess, _ = np.linalg.qr(eigensolver.vectors[:, :roots])

            guess = np.vstack((guess, np.zeros((self.block['size'][1], guess.shape[1]))))

            return guess

        self.guess = guess

        def diagonal(self):
            #EOM-CCSD matrix diagonal preconditioner for Davidson

           return self.reduce(deltas[0], deltas[1])

        self.diagonal = diagonal

        def matvec(self, eom):
            #Construct the matrix-vector product of CCSD similarity-transformed Hamiltonian and
            #the EOM-CCSD right linear excitation operator

            R, EOM= [None, None], self.expand(eom)

            #ph-ph block
            R[0] = -np.einsum('mi,ma->ia', h[0][o, o], EOM[0], optimize=True)
            R[0] += np.einsum('ae,ie->ia', h[0][v, v], EOM[0], optimize=True)
            R[0] += np.einsum('amie,me->ia', h[1][v, o, o, v], EOM[0], optimize=True)
            R[0] -= 0.5 * np.einsum('mnif,mnaf->ia', h[1][o, o, o, v], EOM[1], optimize=True)
            R[0] += 0.5 * np.einsum('anef,inef->ia', h[1][v, o, v, v], EOM[1], optimize=True)
            R[0] += np.einsum('me,imae->ia', h[0][o, v], EOM[1], optimize=True)

            #2ph-2ph block
            R[1] = -0.5 * np.einsum('mi,mjab->ijab', h[0][o, o], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('ae,ijeb->ijab', h[0][v, v], EOM[1], optimize=True)
            R[1] += 0.5 * 0.25 * np.einsum('mnij,mnab->ijab', h[1][o, o, o, o], EOM[1], optimize=True)
            R[1] += 0.5 * 0.25 * np.einsum('abef,ijef->ijab', h[1][v, v, v, v], EOM[1], optimize=True)
            R[1] += np.einsum('amie,mjeb->ijab', h[1][v, o, o, v], EOM[1], optimize=True)
            R[1] -= 0.5 * np.einsum('bmji,ma->ijab', h[1][v, o, o, o], EOM[0], optimize=True)
            R[1] += 0.5 * np.einsum('baje,ie->ijab', h[1][v, v, o, v], EOM[0], optimize=True)

            t = -0.5 * np.einsum('mnef,mnbf->eb', h[1][o, o, v, v], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('eb,ijae->ijab', t, td, optimize=True)

            t = 0.5 * np.einsum('mnef,jnef->mj', h[1][o, o, v, v], EOM[1], optimize=True)
            R[1] -= 0.5 * np.einsum('mj,imab->ijab', t, td, optimize=True)  # A(ij)

            t = np.einsum('amfe,me->af', h[1][v, o, v, v], EOM[0], optimize=True)
            R[1] += 0.5 * np.einsum('af,ijfb->ijab', t, td, optimize=True)  # A(ab)
            t = np.einsum('nmie,me->ni', h[1][o, o, o, v], EOM[0], optimize=True)
            R[1] -= 0.5 * np.einsum('ni,njab->ijab', t, td, optimize=True)  # A(ij)

            #ij and ab permutation
            R[1] = R[1] - R[1].transpose(0,1,3,2) - R[1].transpose(1,0,2,3) + R[1].transpose(1,0,3,2)

            return self.reduce(R[0], R[1])

        self.matrix = matvec

        def r_zero(i):

            r = self.expand(self.cache['vectors'][:, i])
            r0 = np.einsum("me,me->", h[0][o, v], r[0])
            r0 += 0.25 * np.einsum("mnef,mnef->", h[1][o, o, v, v], r[1])

            return r0/self.cache['values'][i]

        eigensolver.roots = self.roots

        #perform direct solve on dense matrix
        if solution == 'direct':
            hamiltonian = self.condense_matrix()
            eigensolver.roots = -1
            eigensolver.direct(hamiltonian)

            self.cache['values']  = eigensolver.values[:self.roots]
            self.cache['vectors'] = eigensolver.vectors[:, :self.roots]

        #perform iterative solution targeting individual roots
        if solution == 'iterative':
            values, vectors = np.zeros(self.roots), np.zeros((sum(self.block['size']), self.roots))

            for i in range(self.roots):
                eigensolver.iterative(self, i)
                if eigensolver.converged:
                    values[i], vectors[:, i] = eigensolver.values, eigensolver.vectors

            #transfer sorted eigensolution to class cache
            idx = np.argsort(values)
            self.cache['values'] = values[idx]
            self.cache['vectors'] = vectors[:, idx]

        self.cache['quasi-weights'] = ([np.linalg.norm(self.cache['vectors'][:, n][:self.block['size'][0]])**2.0
                                       for n, x in enumerate(self.cache['values'])])
        self.cache['r zero'] = [r_zero(n) for n in range(self.roots)]

    def _ip(self, eigensolver, left=False, roots=3, solution='iterative'):

        self.method, self.cache = 'ip', {}
        self.roots  = roots

        #shortform globals
        o, v, nocc, nvir = self.cc.o, self.cc.v, self.nocc, self.nvir
        td, h, deltas = self.td, self.h, self.deltas

        self.matrix_partition_information()

        def guess(self, roots):
            #initial guess Koopmans

            guess = np.zeros((sum(self.block['size']), roots))

            for n in range(roots):
                guess[nocc-n-1, n] = 1.0

            return guess

        self.guess = guess

        def diagonal(self):
            #EOM-CCSD matrix diagonal preconditioner for Davidson IP

            R = [-np.diag(h[0][o, o]), np.zeros((nocc,nocc,nvir))]

            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        R[1][i,j,a] += h[0][v, v][a, a]
                        R[1][i,j,a] += -h[0][o, o][i, i]
                        R[1][i,j,a] += -h[0][o, o][j, j]
                        R[1][i,j,a] += 0.5*(h[1][o, o, o, o][i,j,i,j] - h[1][o, o, o, o][j,i,i,j])
                        R[1][i,j,a] += h[1][v, o, o, v][a,i,i,a]
                        R[1][i,j,a] += h[1][v, o, o, v][a,j,j,a]
                        R[1][i,j,a] += 0.5*(np.dot(h[1][o, o, v, v][i,j,:,a], td[i,j,a,:]) -
                                            np.dot(h[1][o, o, v, v][j,i,:,a], td[i,j,a,:]))

            return self.reduce(R[0], R[1])

        self.diagonal = diagonal

        def matvec(self, eom):
            #Construct the matrix-vector product of CCSD similarity-transformed Hamiltonian and
            #the EOM-CCSD right linear ionisation potential operator

            R, EOM= [None, None], self.expand(eom)

            #1h block
            R[0]  =      -np.einsum('mi,m->i', h[0][o, o], EOM[0], optimize=True)
            R[0] +=       np.einsum('me,mie->i', h[0][o, v], EOM[1], optimize=True)
            R[0] -= 0.5 * np.einsum('nmie,mne->i', h[1][o, o, o, v], EOM[1], optimize=True)

            #2h-1p
            R[1]  =      -np.einsum('amij,m->ija', h[1][v, o, o, o], EOM[0], optimize=True)
            R[1] +=       np.einsum('ae,ije->ija', h[0][v, v], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('mnji,nma->ija', h[1][o, o, o, o], EOM[1], optimize=True)
            R[1] -= 0.5 * np.einsum('mnef,nmf,jiea->ija', h[1][o, o, v, v], EOM[1], td, optimize=True)

            t = -np.einsum('mj,ima->ija', h[0][o, o], EOM[1], optimize=True)
            R[1] += t - t.transpose(1,0,2)

            t =  np.einsum('amie,mje->ija', h[1][v, o, o, v], EOM[1], optimize=True)
            R[1] += t - t.transpose(1,0,2)

            return self.reduce(R[0], R[1])

        def matvec_left(self, eom):
            #Construct the matrix-vector product of CCSD similarity-transformed Hamiltonian and
            #the EOM-CCSD left linear ionisation potential operator

            R, EOM= [None, None], self.expand(eom)

            #1h block
            R[0]  =      -np.einsum('mi,i->m', h[0][o, o], EOM[0], optimize=True)
            R[0] += 0.5 * np.einsum('amji,ija->m', h[1][v, o, o, o], EOM[1], optimize=True)

            #2h-1p
            R[1]  =       np.einsum('me,i->mie', h[0][o, v], EOM[0], optimize=True)
            R[1] -=       np.einsum('ie,m->mie', h[0][o, v], EOM[0], optimize=True)
            R[1] -=       np.einsum('nmie,i->mne', h[1][o, o, o, v], EOM[0], optimize=True)
            R[1] +=       np.einsum('ae,ija->ije', h[0][v, v], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('mnij,ija->mna', h[1][o, o, o, o], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('mnef,ija,ijae->mnf', h[1][o, o, v, v], EOM[1], td, optimize=True)

            t = -np.einsum('mi,ija->mja', h[0][o, o], EOM[1], optimize=True)
            R[1] += t - t.transpose(1,0,2)

            t = np.einsum('amie,ija->mje', h[1][v, o, o, v], EOM[1], optimize=True)
            R[1] += t - t.transpose(1,0,2)

            return self.reduce(R[0], R[1])

        self.matrix = matvec_left if (solution == 'iterative') and left else matvec

        #for Koopmans guess limit roots to available orbitals
        self.roots = min(nocc, self.roots)

        #perform direct solve on dense matrix
        if solution == 'direct':
            hamiltonian = self.condense_matrix()
            eigensolver.roots = -1
            eigensolver.direct(hamiltonian, left)

            self.cache['values'] = eigensolver.values[:self.roots]
            self.cache['vectors'] = eigensolver.vectors[:, :self.roots]

        if solution == 'iterative':
            values, vectors = np.zeros(self.roots), np.zeros((sum(self.block['size']), self.roots))
            for i in range(self.roots):
                eigensolver.iterative(self, i)
                if eigensolver.converged:
                    values[i], vectors[:, i] = eigensolver.values, eigensolver.vectors

            #transfer sorted eigensolution to class cache
            idx = np.argsort(values)
            self.cache['values'] = values[idx]
            self.cache['vectors'] = vectors[:, idx]

        self.cache['quasi-weights'] = ([np.linalg.norm(self.cache['vectors'][:, n][:self.block['size'][0]])**2.0
                                       for n, x in enumerate(self.cache['values'])])

    def _ea(self, eigensolver, left=False, roots=3, solution='iterative'):

        self.method, self.cache = 'ea', {}
        self.roots  = roots

        #shortform globals
        o, v, nocc, nvir = self.cc.o, self.cc.v, self.nocc, self.nvir
        td, h, deltas = self.td, self.h, self.deltas

        self.matrix_partition_information()

        def guess(self, roots):
            #initial guess Koopmans

            guess = np.zeros((sum(self.block['size']), roots))

            for n in range(roots):
                guess[n, n] = 1.0

            return guess

        self.guess = guess

        def diagonal(self):
            #EOM-CCSD matrix diagonal preconditioner for Davidson IP

            R = [-np.diag(h[0][v, v]), np.zeros((nocc,nvir,nvir))]

            for i in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        R[1][i,a,b] += h[0][v, v][a, a]
                        R[1][i,a,b] += h[0][v, v][b, b]
                        R[1][i,a,b] -= h[0][o, o][i, i]
                        R[1][i,a,b] += 0.5*(h[1][v, v, v, v][a,b,a,b] - h[1][v, v, v, v][b,a,a,b])
                        R[1][i,a,b] += h[1][v, o, o, v][b,i,i,b]
                        R[1][i,a,b] += h[1][v, o, o, v][a,i,i,a]
                        R[1][i,a,b] -= 0.5*(np.dot(h[1][o, o, v, v][:,i,a,b], td[:,i,a,b]) -
                                            np.dot(h[1][o, o, v, v][:,i,b,a], td[:,i,a,b]))

            return self.reduce(R[0], R[1])

        self.diagonal = diagonal

        def matvec(self, eom):
            #Construct the matrix-vector product of CCSD similarity-transformed Hamiltonian and
            #the EOM-CCSD right linear electron affinity operator

            R, EOM= [None, None], self.expand(eom)

            #1p block
            R[0]  =       np.einsum('ae,e->a', h[0][v, v], EOM[0], optimize=True)
            R[0] +=       np.einsum('me,mae->a', h[0][o, v], EOM[1], optimize=True)
            R[0] += 0.5 * np.einsum('anef,nef->a', h[1][v, o, v, v], EOM[1], optimize=True)

            #ih-2p
            R[1]  =       np.einsum('baie,e->iab', h[1][v, v, o, v], EOM[0], optimize=True)
            R[1] -=       np.einsum('mi,mab->iab', h[0][o, o], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('abef,ief->iab', h[1][v, v, v, v], EOM[1], optimize=True)
            R[1] -= 0.5 * np.einsum('mnef,nef,miab->iab', h[1][o, o, v, v], EOM[1], td, optimize=True)

            t = np.einsum('ae,ieb->iab', h[0][v, v], EOM[1], optimize=True)
            R[1] += t - t.transpose(0,2,1)

            t = np.einsum('bmie,mae->iab', h[1][v, o, o, v], EOM[1], optimize=True)
            R[1] += t - t.transpose(0,2,1)

            return self.reduce(R[0], R[1])

        def matvec_left(self, eom):
            #Construct the matrix-vector product of CCSD similarity-transformed Hamiltonian and
            #the EOM-CCSD left linear electron attachment operator

            R, EOM= [None, None], self.expand(eom)

            #1p block
            R[0]  =       np.einsum('ae,a->e', h[0][v, v], EOM[0], optimize=True)
            R[0] -= 0.5 * np.einsum('abje,jab->e', h[1][v, v, o, v], EOM[1], optimize=True)

            #1h-2p
            R[1]  =       np.einsum('amef,a->mef', h[1][v, o, v, v], EOM[0], optimize=True)
            R[1] +=       np.einsum('mf,e->mef', h[0][o, v], EOM[0], optimize=True)
            R[1] -=       np.einsum('me,f->mef', h[0][o, v], EOM[0], optimize=True)

            R[1] -=       np.einsum('mi,ief->mef', h[0][o, o], EOM[1], optimize=True)
            R[1] += 0.5 * np.einsum('abef,mab->mef', h[1][v, v, v, v], EOM[1], optimize=True)
            R[1] -= 0.5 * np.einsum('imef,jab,ijab->mef', h[1][o, o, v, v], EOM[1], td, optimize=True)

            t = np.einsum('ae,maf->mef', h[0][v, v], EOM[1], optimize=True)
            R[1] += t - t.transpose(0,2,1)

            t = np.einsum('amif,iea->mef', h[1][v, o, o, v], EOM[1], optimize=True)
            R[1] += t - t.transpose(0,2,1)

            return self.reduce(R[0], R[1])

        self.matrix = matvec_left if (solution == 'iterative') and left else matvec

        #perform direct solve on dense matrix
        if solution == 'direct':
            hamiltonian = self.condense_matrix()
            eigensolver.roots = -1
            eigensolver.direct(hamiltonian, left)

            self.cache['values'] = eigensolver.values[:self.roots]
            self.cache['vectors'] = eigensolver.vectors[:, :self.roots]

        if solution == 'iterative':
            values, vectors = np.zeros(self.roots), np.zeros((sum(self.block['size']), self.roots))
            for i in range(self.roots):
                eigensolver.iterative(self, i)
                if eigensolver.converged:
                    values[i], vectors[:, i] = eigensolver.values, eigensolver.vectors

            #transfer sorted eigensolution to class cache
            idx = np.argsort(values)
            self.cache['values'] = values[idx]
            self.cache['vectors'] = vectors[:, idx]

        self.cache['quasi-weights'] = ([np.linalg.norm(self.cache['vectors'][:, n][:self.block['size'][0]])**2.0
                                       for n, x in enumerate(self.cache['values'])])
    def analyse(self):
        #analyse the eigenvectors

        def components(u):
            #return the spin-components of eigenvector u
            if u.ndim == 1:
                return [u[::2].ravel(), u[1::2].ravel()]
            if u.ndim == 2:
                return [u[::2,::2].ravel(),  u[1::2,1::2].ravel(),
                        u[::2,1::2].ravel(), u[1::2,::2].ravel()]
            if u.ndim == 3:
                return [u[::2,::2,::2].ravel(),  u[1::2,1::2,1::2].ravel(),
                        u[::2,1::2,::2].ravel(), u[1::2,::2,1::2].ravel()]
            if u.ndim == 4:
                return [u[::2,::2,::2,::2].ravel(),   u[1::2,1::2,1::2,1::2].ravel(),
                        u[::2,1::2,::2,1::2].ravel(), u[1::2,::2,1::2,::2].ravel()]

        #spin symbols
        symbol = ['\u2963', '\u296E']

        #convert eigenvectors to amplitudes
        vectors = [self.cache['vectors'][:, n] for n in range(self.roots)]

        #cache dictionary lists
        self.cache['norms'], self.cache['excitations'] = [], []

        for root in range(self.roots):

            v, u = vectors[root][:self.block['size'][0]], vectors[root][self.block['size'][0]:]

            self.cache['norms'].append([float(np.einsum('i,i->', v, v, optimize=True)),
                                        float(np.einsum('i,i->', u, u, optimize=True))])

            #1p or 1h block excitations
            u = (self.cache['vectors'][:self.block['size'][0], root]**2.0).reshape(self.block['shape'][0])
            x = np.sqrt(sum(components(u))) if self.method in ['ip', 'ea'] else np.sqrt(sum(components(u))) / pow(2, 0.5)

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
                spin_symbol = '' if self.method in ['ip', 'ea'] else symbol[spin]

                excitations = []
                #add state information - corrected for orbital indexing from 1
                if ix != []:
                    bias = self.block['orbital'][0]

                    if not self.options['LEADING'] is None:
                        effective_leading = min(dominant_element_count, self.LEADING)
                        x, ix = x[:effective_leading], ix[:effective_leading]

                    ix_orbitals = []
                    for i in ix:
                        ix_orbitals.append([j+bias[n] for n, j in enumerate(i)])

                    excitations.append([self.block['label'][0], list(x), ix_orbitals, spin_symbol])

                self.cache['excitations'].append(excitations)
