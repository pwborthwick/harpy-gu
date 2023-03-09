from __future__ import division
import scipy as sp
import numpy as np

'''
scipy linalg - [https://docs.scipy.org/doc/scipy/reference/linalg.html]
numpy linalg - [https://numpy.org/doc/stable/reference/routines.linalg.html]
Davidson - [https://www.mat.tuhh.de/lehre/material/ewa/chap3_ho.pdf] [https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter12.pdf]
'''

class solver(object):
    #eigensolution

    def __init__(self, roots=-1, vectors=True, cycles=50, sort_on_absolute=True, tol=1e-8, vectors_per_root=30):

        #roots = -1 => all roots, -n => decending order, +n => ascending order
        self.roots = roots
        self.do_vectors = vectors

        #Davidson parameters
        self.cycles, self.sort_absolute, self.tol, self.vectors_per_root = cycles, sort_on_absolute, tol, vectors_per_root

    def decending(values, vectors):
        #reverse order of eigen solution

        idx = values.argsort()[::-1]

        return values[idx], vectors[:, idx]


    def direct(self, matrix, left=True):
        #direct scipy/numpy eigensolvers despatcher

        self.matrix = matrix

        #check symmetry
        self.symmetric_solve() if np.allclose(matrix, matrix.transpose()) else self.asymmetric_solve(left)

    def iterative(self, object, root=None):
        #davidson despatcher

        if self.roots == -1: self.roots = 3

        if root is None:
            self.values, self.vectors, self.converged = self.davidson(object.matrix, object.guess(object, self.roots),
                                                                      object.diagonal(object), self.cycles,
                                                                      self.sort_absolute, self.tol, self.vectors_per_root)

            self.values, self.vectors = self.values[:self.roots], self.vectors[:, :self.roots]
        else:
            self.values, self.vectors, self.converged = self.davidson_targeted(object, root, self.cycles, self.tol)

    def symmetric_solve(self):
        #use scipy symmetric eigensolver

        self.local_roots = abs(self.roots) if self.roots != -1 else self.matrix.shape[0]

        try:
            if self.do_vectors:
                eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
                if self.roots < -1:
                    eigenvalues, eigenvectors = solver.decending(eigenvalues, eigenvectors)

                self.values, self.vectors = eigenvalues[:self.local_roots], eigenvectors[:, :self.local_roots]
            else:
                eigenvalues = sp.linalg.eigvalsh(self.matrix)
                if self.roots < -1:
                    eigenvalues, eigenvectors = solver.decending(eigenvalues, eigenvectors)

                self.values, self.vectors = eigenvalues[:self.local_roots], None
            self.converged = True

        except np.linalg.LinAlgError as script:
            print('Linear algebra error :', script)
            self.converged = False


    def asymmetric_solve(self, left=False):
        #use scipy general eigensolver

        self.local_roots = abs(self.roots) if self.roots != -1 else self.matrix.shape[0]
        eig_range =  (0, self.roots-1)

        try:
            if self.do_vectors:
                right = not left
                eigenvalues, eigenvectors = sp.linalg.eig(self.matrix, right=right, left=left)
                values, vectors = eigenvalues, eigenvectors

                #order ascending
                idx = np.argsort(values)
                if self.roots < -1: idx = idx[::-1]
                self.values, self.vectors = values[idx][:self.local_roots], vectors[:, idx][:, :self.local_roots]
            else:
                eigenvalues = sp.linalg.eigvals(self.matrix)

                #order ascending
                idx = np.argsort(eigenvalues)
                if self.roots < -1: idx = idx[::-1]
                self.values, self.vectors = eigenvalues[idx][:self.local_roots], None
            self.converged = True

        except np.linalg.LinAlgError as script:
            print('Linear algebra error :', script)
            self.converged = False

        #convert to real if imaginary residuals
        if self.converged and (self.values.dtype == np.dtype('complex128')):
            self.values  = np.real_if_close(self.values,  tol=1)
            self.vectors = np.real_if_close(self.vectors, tol=1)

    def davidson(self, matrix, guess_subspace, diagonal, cycles=50, sort_on_absolute=True, tol=1e-8, vectors_per_root=30):
        #asymmetric Davidson iterator
        #this code is based on O Backhouse's solver in psi4numpy (ADC_helper)

        converged = False

        if callable(matrix):
            matvec = matrix
        else:
            matvec = lambda _, x: np.dot(matrix, x)

        if sort_on_absolute:
            selector = lambda x: np.argsort(np.absolute(x))[:k]
        else:
            selector = lambda x: np.argsort(x)[:k]

        k = guess_subspace.shape[1]
        b = guess_subspace.copy()
        theta = np.zeros((k,))

        if cycles is None:
            cycles = k * 15

        for cycle in range(cycles):

            #orthogonalize sub-space set
            b, r = np.linalg.qr(b)
            ex_theta = theta[:k]

            #solve sub-space Hamiltonian
            s = np.zeros_like(b)
            for i in range(b.shape[1]):
                s[:,i] = matvec(self, b[:,i])

            g = np.dot(b.T, s)
            theta, S = np.linalg.eigh(g)

            #selected biggest eigenvalues (theta) and corresponding vectors (S)
            idx = selector(theta)
            theta = theta[idx]
            S = S[:,idx]

            #augment sub-space
            b_augment = []
            for i in range(k):
                w  = np.dot(s, S[:,i])
                w -= np.dot(b, S[:,i]) * theta[i]
                q = w / (theta[i] - diagonal[i] + 1e-30)
                b_augment.append(q)

            #check for converged roots
            if np.linalg.norm(theta[:k] - ex_theta) < tol:
                converged = True
                b = np.dot(b, S)
                break

            else:
                #collapse sub-space if too large or augment sub-space
                if b.shape[1] >= (k * vectors_per_root):
                    b = np.dot(b, S)
                    theta = ex_theta
                else:
                    b = np.hstack([b, np.column_stack(b_augment)])

        b = b[:, :k]

        return theta, b, converged

    def davidson_targeted(self, object, root, cycles=50, tol=1e-8):

        # Initial values
        b = object.guess(object, object.roots)[:, root].reshape(-1, 1)
        s = object.matrix(object, b[:, 0]).reshape(-1, 1)
        theta = 0.0

        converged = False

        for cycle in range(1, cycles):

            #last iteration energy
            ex_theta = theta

            #solve projected subspace
            g = np.dot(b.T, s)
            theta, S = np.linalg.eig(g)

            #column with maximum overlap with initial guess
            idx = np.argsort(abs(S[0, :]))[-1]

            #get selected eigenpair
            S = np.real(S[:, idx])
            theta = np.real(theta[idx])

            # calculate residual vector
            w = np.dot(s, S)
            w -= theta * np.dot(b, S)

            #pre-condition
            q = w/(theta - object.diagonal(object) + 1e-30)

            #append new vector using Gram-Schmidt orthogonalisation
            for p in range(cycle):
                v = b[:, p] * np.reciprocal(np.linalg.norm(b[:, p]))
                q -= np.dot(v.T, q) * v
            q *= np.reciprocal(np.linalg.norm(q))

            delta_e = theta - ex_theta

            if np.linalg.norm(w) < tol and abs(delta_e) < tol:
                converged = True
                break

            b = np.concatenate([b, q.reshape(b.shape[0], 1)], axis=1)
            s = np.concatenate([s, object.matrix(object, b[:, cycle]).reshape(b.shape[0], 1)], axis=1)

        return theta, np.dot(b, S), converged
