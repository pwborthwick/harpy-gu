from __future__ import division
import numpy as np

'''
Crawford Projects 8 - [https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2308]
'''

class dii_subspace(object):
    #direct inversion of iterative subspace class

    def __init__(self, size, type='f'):
        self.size = size
        self.cache  = []
        self.error_vector = []
        self.norm = 0.0

        self.append = self.append_f if type == 'f' else self.append_c
        self.build  = self.build_f  if type == 'f' else self.build_c

    def append_f(self, f, d, s, x):
        #update the subspaces respecting size of cache

        if f.ndim == 3:
            #spin-polarised computation
            s = (s,s)
            g = []
            for ni in range(f.ndim-1):
                fds = np.einsum('im,mn,nj->ij',f[ni], d[ni], s[ni], optimize=True)
                g.append(np.einsum('mi,mn,nj->ij', x, (fds - fds.T), x, optimize=True))

            self.error_vector.append(np.vstack(g))
            self.cache.append(f)
        else:
            #spin-unpolarised computation
            self.cache.append(f) 
            fds = np.einsum('im,mn,nj->ij',f, d, s, optimize=True)
            self.error_vector.append(np.einsum('mi,mn,nj->ij', x, (fds - fds.T), x, optimize=True))

        self.norm = np.linalg.norm(self.error_vector[-1])

        #check size
        if len(self.cache) > self.size:
            del self.cache[0]
            del self.error_vector[0]


    def build_f(self, f, d, s, x):
        #compute extrapolated Fock

        #update caches
        self.append(f, d, s, x)

        #construct B matrix
        nSubSpace = len(self.cache)
        
        #start diis after cache full
        if nSubSpace < self.size: return f

        b = -np.ones((nSubSpace+1,nSubSpace+1))
        b[:-1,:-1] = 0.0 ; b[-1,-1] = 0.0
        for i in range(nSubSpace):
            for j in range(nSubSpace):
                b[i,j] = b[j,i] = np.einsum('ij,ij->',self.error_vector[i], self.error_vector[j], optimize=True)


        #solve for weights
        residuals = np.zeros(nSubSpace+1)
        residuals[-1] = -1

        try:
            weights = np.linalg.solve(b, residuals)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e): exit('diis failed with singular matrix')

        #weights should sum to +1
        sum = np.sum(weights[:-1])
        assert np.isclose(sum, 1.0)

        #construct extrapolated Fock
        f = np.zeros_like(f, dtype='float')
        for i in range(nSubSpace):
            f += self.cache[i] * weights[i]

        if f.ndim == 3:
            f = f.reshape(d.shape)

        return f

    def append_c(self, amplitudes):
        #store current amplitude prior to update

        self.store = amplitudes

    def build_c(self, amplitudes):
        #build the new amplitudes

        self.cache.append([i.copy() for i in amplitudes])
        n_amplitudes = len(amplitudes)
        error = [(self.cache[-1][i] - self.store[i]).ravel() for i in range(n_amplitudes)]
        self.error_vector.append(np.concatenate(error))

        self.norm = np.linalg.norm(self.error_vector[-1])

        if (len(self.cache) > self.size):
            del self.cache[0]
            del self.error_vector[0]
        nSubSpace = len(self.cache) - 1

        #construct b-matrix
        b = -np.ones((nSubSpace+1, nSubSpace+1))
        b[-1, -1] = 0

        b = np.zeros((nSubSpace+1,nSubSpace+1))
        for i in range(0, nSubSpace):
            for j in range(0, i+1):
                b[j,i] = np.dot(self.error_vector[i], self.error_vector[j])
                b[i,j] = b[j,i]
        for i in range(0, nSubSpace):
            b[nSubSpace, i] = -1
            b[i, nSubSpace] = -1

        # Build residual vector
        residual = np.zeros(nSubSpace+1)
        residual[-1] = -1

        # Solve Pulay equations for weights
        try:
            w = np.linalg.solve(b, residual)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e): exit('diis failed with singular matrix')

        # Calculate new amplitudes
        amplitudes = [amplitudes[i]*0.0 for i in range(n_amplitudes)]
        for num in range(nSubSpace):
            for i in range(n_amplitudes): amplitudes[i] += w[num] * self.cache[num + 1][i]

        return amplitudes
