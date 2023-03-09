from __future__ import division
import int.mo_spin as mos

import numpy as np

'''
Szabo and Ostlund - Modern Quantum Chemistry chapter 7
One-particle many-body Green’s function theory: Algebraic recursive definitions, linked-diagram theorem, 
irreducible-diagram theorem, and general-order algorithms - So Hirata, Alexander E. Doran, Peter J. Knowles, and J. V. Ortiz
 The Journal of Chemical Physics 147, 044108 (2017) - [https://www.osti.gov/pages/servlets/purl/1473852]
'''
class EP(object):
    '''
    class for electron propagator - self-energy one-electron Green's function
    '''

    def __init__(self, scf, orbital_range='HOMO-2, LUMO+1'):
        #orbital range is given from base zero in spatial count
        
        self.scf = scf

        self.build_mo_spin_environment()
        self.decode_range(orbital_range.upper())

        self.scf.tol = 1e-6
        
    def build_mo_spin_environment(self):
        '''
        transform orbital energies and 2-electron repulsion integrals to molecular
        spin basis
        '''

        self.eps = mos.orbital_transform(self.scf, 'm+s', self.scf.get('e'))
        self.gs  = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))

        #occupations and slices
        self.nmo, self.nocc = self.scf.mol.norb*2, sum(self.scf.mol.nele)
        self.o, self.v = slice(None, self.nocc), slice(self.nocc, None)

    def decode_range(self, orbital_range):
        '''
        convert string description of orbital range to numeric list
        '''

        ndocc = self.nocc//2
        homo, lumo = ndocc - 1, ndocc
        orbital_range = orbital_range.replace('HOMO', str(homo)).replace('LUMO', str(lumo)).split(',')

        #eval strings and ensure orbitals in valid range
        self.orbital_range = [eval(x) for x in orbital_range]
        self.orbital_range[0], self.orbital_range[1] = (max(0, self.orbital_range[0]),
                                                        min(self.scf.mol.norb - 1, self.orbital_range[1]))
        
    def method(self, order):
        '''
        dispatcher for electron propagator methods
        '''

        if not order in [2, 3]:
            print('Electron propagator for orders 2 and 3 only')
            return
        
        if order == 2: self.electron_propagator_order_2()
        if order == 3: self.electron_propagator_order_3()
        
    def electron_propagator_order_2(self):
        '''
        solve E = \epsilon_p + \Sigma^{(2)}_{pp} by Newton-Raphson method in molecular
        spin basis
        '''

        cache = []
        o, v, n = self.o, self.v, np.newaxis
        
        #loop over orbitals requested
        for p in range(2 * self.orbital_range[0], 2 * self.orbital_range[1] + 1, 2):

            #diagonal approximation 
            q = p

            omega = self.eps[p]
            self.converged = False
            
            cycle_energy, delta, sigma, dsigma = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
            for cycle in range(self.scf.cycles):

                cycle_energy[0] = omega

                #build orbital energy differences
                delta[0] = np.reciprocal(omega + self.eps[o,n,n] - self.eps[n,v,n] - self.eps[n,n,v])
                delta[1] = np.reciprocal(self.eps[o,n,n] + self.eps[n,o,n] - omega - self.eps[n,n,v])
                
                #build sigmas(2)
                sigma[0] =  0.5 * np.einsum('iab,abi,iab->', self.gs[o,q,v,v], self.gs[v,v,o,p], delta[0], optimize=True)
                sigma[1] = -0.5 * np.einsum('aij,ija,ija->', self.gs[v,q,o,o], self.gs[o,o,v,p], delta[1], optimize=True)

                cycle_energy[1] = self.eps[p] + sigma[0] + sigma[1]

                #check for convergence
                if abs(cycle_energy[1] - cycle_energy[0]) < self.scf.tol:
                    self.converged = True
                    cache.append({'sigma': cycle_energy[1], 'koo': self.eps[p]})

                    break

                #build derivatives
                dsigma[0] = -0.5 * np.einsum('iab,abi,iab->', self.gs[o,q,v,v], self.gs[v,v,o,p], np.power(delta[0], 2), optimize=True)
                dsigma[1] = -0.5 * np.einsum('aij,ija,ija->', self.gs[v,q,o,o], self.gs[o,o,v,p], np.power(delta[1], 2), optimize=True)
                derivative = 1.0 - (dsigma[0] + dsigma[1])

                #Newton-Raphson update
                omega = cycle_energy[0] - (cycle_energy[0] - cycle_energy[1]) / derivative

            if not self.converged:
                cache.append({'sigma': cycle_energy[1], 'koo': self.eps[p]})
                print('***warn - EP2 for orbital [', p, '] did not converge ***')

        self.cache = cache
        return
        
    def electron_propagator_order_3(self):
        '''
        solve E = \epsilon_p + \Sigma^{(2)}_{pp} + \Sigma^{(3)}_{pp} by Newton-Raphson method in molecular
        spin basis
        '''

        cache = []
        o, v, n = self.o, self.v, np.newaxis

        orbital_range = self.orbital_range
        
        #loop over orbitals requested
        for p in range(2 * self.orbital_range[0], 2 * self.orbital_range[1] + 1, 2):

            #diagonal approximation 
            q = p

            #compute sigma second order for inital guess
            self.orbital_range = [p//2, p//2]
            self.electron_propagator_order_2()
            omega = self.cache[0]['sigma']

            sigma, dsigma = [0.0 for i in range(20)], [0.0 for i in range(20)]
            delta = [0.0 for i in range(4)]
            
            #energy independent - self.energy bubble diagrams
            delta[0] = np.reciprocal(self.eps[o,n,n,n] + self.eps[n,o,n,n] - self.eps[n,n,v,n] - self.eps[n,n,n,v])
            delta[1] = np.reciprocal(self.eps[o,n] - self.eps[n,v])
            
            sigma[0] =  0.5 * np.einsum('abij,ea,ijeb,ijab,ijbe->', self.gs[v,v,o,o], self.gs[p,v,q,v], self.gs[o,o,v,v],
                                                                    delta[0], delta[0], optimize=True)
            sigma[1] = -0.5 * np.einsum('mjab,im,abij,ijab,jmab->', self.gs[o,o,v,v], self.gs[p,o,q,o], self.gs[v,v,o,o],
                                                                    delta[0], delta[0], optimize=True)
            sigma[2] =  0.5 * np.einsum('ijbe,ai,beaj,ijbe,ia->', self.gs[o,o,v,v], self.gs[p,v,q,o], self.gs[v,v,v,o],
                                                                    delta[0], delta[1], optimize=True)
            sigma[3] = -0.5 * np.einsum('jmab,ai,ibjm,jmab,ia->', self.gs[o,o,v,v], self.gs[p,v,q,o], self.gs[o,v,o,o],
                                                                    delta[0], delta[1], optimize=True)
            sigma[4] =  0.5 * np.einsum('ejab,ie,abij,ijab,ie->', self.gs[v,o,v,v], self.gs[p,o,q,v], self.gs[v,v,o,o],
                                                                    delta[0], delta[1], optimize=True)
            sigma[5] = -0.5 * np.einsum('abij,ma,ijmb,ijab,ma->', self.gs[v,v,o,o], self.gs[p,o,q,v], self.gs[o,o,o,v],
                                                                    delta[0], delta[1], optimize=True)
            
            self.converged = False

            cycle_energy = [0.0, 0.0]
            for cycle in range(self.scf.cycles):

                cycle_energy[0] = omega
                
                #self-energy diagrams and derivative for order 3
                delta[2] = np.reciprocal(omega + self.eps[o,n,n] - self.eps[n,v,n] - self.eps[n,n,v])
                delta[3] = np.reciprocal(omega + self.eps[v,n,n] - self.eps[n,o,n] - self.eps[n,n,o])

                #Hugenholtz diagrams of order 3 with an outer line cut - equivalent lines
                sigma[6]  =  0.25 * np.einsum('iab,abef,efi,ief,iab->', self.gs[p,o,v,v], self.gs[v,v,v,v], self.gs[v,v,q,o],
                                                                      delta[2], delta[2], optimize=True)
                sigma[7]  = -0.25 * np.einsum('aij,ijmn,mna,amn,aij->', self.gs[p,v,o,o], self.gs[o,o,o,o], self.gs[o,o,q,v],
                                                                      delta[3], delta[3], optimize=True)
                #Hugenholtz diagrams of order 3 with an inner line cut - equivalent lines
                sigma[8]  =  0.25 * np.einsum('mij,ijab,abm,mab,ijab->', self.gs[p,o,o,o], self.gs[o,o,v,v], self.gs[v,v,q,o],
                                                                      delta[2], delta[0], optimize=True)
                sigma[9]  =  0.25 * np.einsum('iab,abjm,jmi,iab,jmab->', self.gs[p,o,v,v], self.gs[v,v,o,o], self.gs[o,o,q,o],
                                                                      delta[2], delta[0], optimize=True)
                sigma[10] =  0.25 * np.einsum('aij,ijbe,bea,aij,ijbe->', self.gs[p,v,o,o], self.gs[o,o,v,v], self.gs[v,v,q,v],
                                                                      delta[3], delta[0], optimize=True)
                sigma[11] =  0.25 * np.einsum('abe,beij,ija,aij,ijbe->', self.gs[p,v,v,v], self.gs[v,v,o,o], self.gs[o,o,q,v],
                                                                      delta[3], delta[0], optimize=True)
                #Hugenholtz diagrams of order 3 with outer line cut - inequivalent lines
                sigma[12] =        -np.einsum('ibe,bjai,aej,jae,ibe->',  self.gs[p,o,v,v], self.gs[v,o,v,o], self.gs[v,v,q,o],
                                                                      delta[2], delta[2], optimize=True)
                sigma[13] =         np.einsum('ajm,jbia,imb,ajm,bim->', self.gs[p,v,o,o], self.gs[o,v,o,v], self.gs[o,o,q,v],
                                                                      delta[3], delta[3], optimize=True)
                #Hugenholtz diagrams of order 3 with innerr line cut - inequivalent lines                
                sigma[14] =        -np.einsum('eib,ijae,abj,jab,ijae->', self.gs[p,v,o,v], self.gs[o,o,v,v], self.gs[v,v,q,o],
                                                                      delta[2], delta[0], optimize=True)
                sigma[15] =        -np.einsum('jae,abij,ieb,jae,ijab->', self.gs[p,o,v,v], self.gs[v,v,o,o], self.gs[o,v,q,v],
                                                                      delta[2], delta[0], optimize=True)
                sigma[16] =        -np.einsum('bim,ijab,amj,bim,ijab->', self.gs[p,v,o,o], self.gs[o,o,v,v], self.gs[v,o,q,o],
                                                                      delta[3], delta[0], optimize=True)
                sigma[17] =        -np.einsum('maj,abim,ijb,bij,imab->', self.gs[p,o,v,o], self.gs[v,v,o,o], self.gs[o,o,q,v],
                                                                      delta[3], delta[0], optimize=True)
                #Hugenholtz diagrams for order 2
                sigma[18] =   0.5 * np.einsum('iab,abi,iab->', self.gs[p,o,v,v], self.gs[v,v,q,o], delta[2], optimize=True)
                sigma[19] =  -0.5 * np.einsum('aij,ija,aij->', self.gs[p,v,o,o], self.gs[o,o,v,q], delta[3], optimize=True)

                cycle_energy[1] = self.eps[p] + sum(sigma)
                
                #check for convergence
                if abs(cycle_energy[1] - cycle_energy[0]) < self.scf.tol:
                    self.converged = True
                    cache.append({'sigma': cycle_energy[1], 'koopman': self.eps[p]})

                    break

                #sigma derivatives
                dsigma[6]  = -0.25 * (np.einsum('iab,abef,efi,ief,iab->', self.gs[p,o,v,v], self.gs[v,v,v,v], self.gs[v,v,q,o],
                                                                       np.power(delta[2], 2), delta[2], optimize=True)
                                     +np.einsum('iab,abef,efi,ief,iab->', self.gs[p,o,v,v], self.gs[v,v,v,v], self.gs[v,v,q,o],
                                                                       delta[2], np.power(delta[2],2), optimize=True))
                dsigma[7]  =  0.25 * (np.einsum('aij,ijmn,mna,amn,aij->', self.gs[p,v,o,o], self.gs[o,o,o,o], self.gs[o,o,q,v],
                                                                       np.power(delta[3], 2), delta[3], optimize=True)
                                     +np.einsum('aij,ijmn,mna,amn,aij->', self.gs[p,v,o,o], self.gs[o,o,o,o], self.gs[o,o,q,v],
                                                                       delta[3], np.power(delta[3], 2), optimize=True))
                
                dsigma[8]  = -0.25 *  np.einsum('mij,ijab,abm,mab,ijab->', self.gs[p,o,o,o], self.gs[o,o,v,v], self.gs[v,v,q,o],
                                                                       np.power(delta[2], 2), delta[0], optimize=True)
                dsigma[9]  = -0.25 *  np.einsum('iab,abjm,jmi,iab,jmab->', self.gs[p,o,v,v], self.gs[v,v,o,o], self.gs[o,o,q,o],
                                                                       np.power(delta[2], 2), delta[0], optimize=True)
                dsigma[10] = -0.25 *  np.einsum('aij,ijbe,bea,aij,ijbe->', self.gs[p,v,o,o], self.gs[o,o,v,v], self.gs[v,v,q,v],
                                                                       np.power(delta[3], 2), delta[0], optimize=True)
                dsigma[11] = -0.25 *  np.einsum('abe,beij,ija,aij,ijbe->', self.gs[p,v,v,v], self.gs[v,v,o,o], self.gs[o,o,q,v],
                                                                       np.power(delta[3], 2), delta[0], optimize=True)

                dsigma[12] =  (np.einsum('ibe,bjai,aej,jae,ibe->',  self.gs[p,o,v,v], self.gs[v,o,v,o], self.gs[v,v,q,o],
                                                                np.power(delta[2], 2), delta[2], optimize=True)
                              +np.einsum('ibe,bjai,aej,jae,ibe->',  self.gs[p,o,v,v], self.gs[v,o,v,o], self.gs[v,v,q,o],
                                                                delta[2], np.power(delta[2], 2), optimize=True))
                dsigma[13] = -(np.einsum('ajm,jbia,imb,ajm,bim->', self.gs[p,v,o,o], self.gs[o,v,o,v], self.gs[o,o,q,v],
                                                                np.power(delta[3], 2), delta[3], optimize=True)
                              +np.einsum('ajm,jbia,imb,ajm,bim->', self.gs[p,v,o,o], self.gs[o,v,o,v], self.gs[o,o,q,v],
                                                                delta[3], np.power(delta[3], 2), optimize=True))

                dsigma[14] =   np.einsum('eib,ijae,abj,jab,ijae->', self.gs[p,v,o,v], self.gs[o,o,v,v], self.gs[v,v,q,o],
                                                                np.power(delta[2], 2), delta[0], optimize=True)
                dsigma[15] =   np.einsum('jae,abij,ieb,jae,ijab->', self.gs[p,o,v,v], self.gs[v,v,o,o], self.gs[o,v,q,v],
                                                                np.power(delta[2], 2), delta[0], optimize=True)
                dsigma[16] =   np.einsum('bim,ijab,amj,bim,ijab->', self.gs[p,v,o,o], self.gs[o,o,v,v], self.gs[v,o,q,o],
                                                                np.power(delta[3], 2), delta[0], optimize=True)
                dsigma[17] =   np.einsum('maj,abim,ijb,bij,imab->', self.gs[p,o,v,o], self.gs[v,v,o,o], self.gs[o,o,q,v],
                                                                np.power(delta[3], 2), delta[0], optimize=True)
                
                dsigma[18] = -0.5 * np.einsum('iab,abi,iab->', self.gs[p,o,v,v], self.gs[v,v,q,o], np.power(delta[2], 2), optimize=True)
                dsigma[19] = -0.5 * np.einsum('aij,ija,aij->', self.gs[v,q,o,o], self.gs[o,o,v,p], np.power(delta[3], 2), optimize=True)
                
                #Newton-Raphson update
                derivative = 1.0 - sum(dsigma)
                omega = cycle_energy[0] - (cycle_energy[0] - cycle_energy[1]) / derivative


            if not self.converged:
                cache.append({'sigma': cycle_energy[1], 'koopman': self.eps[p]})
                print('***warn - EP2 for orbital [', p, '] did not converge ***')

        #restore class properties altered by external calls
        self.cache = cache
        self.orbital_range = orbital_range

        return
                
    def approximate_greens_function_order_2(self):
        '''
        the approximate Green's function (2) correction to IP
        '''
        
        agf_energies, koopman = [], []
        o, v, n = self.o, self.v, np.newaxis
        
        #loop over orbitals requested - only do IP
        if self.orbital_range[1] > self.scf.mol.nele[0] - 1: self.orbital_range[1] = self.scf.mol.nele[0] - 1

        self.cache = []
        
        for p in range(2 * self.orbital_range[0], 2 * self.orbital_range[1] + 1, 2):

            #diagonal approximation 
            q = p

            omega = self.eps[p]
            
            delta, sigma = [0.0, 0.0, 0.0], [0.0, 0.0]

            #build orbital energy differences
            delta[0] = np.reciprocal(omega + self.eps[o,n,n] - self.eps[n,v,n] - self.eps[n,n,v])
            delta[1] = np.reciprocal(self.eps[o,n,n] + self.eps[n,o,n] - omega - self.eps[n,n,v])
            delta[2] = np.reciprocal(self.eps[o,n] - self.eps[n,v])
            
            #build sigmas(2)
            sigma[0] =  0.5 * np.einsum('iab,abi,iab->', self.gs[o,q,v,v], self.gs[v,v,o,p], delta[0], optimize=True)
            sigma[1] = -0.5 * np.einsum('aij,ija,ija->', self.gs[v,q,o,o], self.gs[o,o,v,p], delta[1], optimize=True)

            #energy components
            relaxation_energy, pair_removal_energy = sigma[1], sigma[0]
            orbital_relaxation_energy = np.einsum('ia,ia->', self.gs[o,p,q,v], delta[2], optimize=True)
            pair_relaxation_energy = relaxation_energy - orbital_relaxation_energy
            koopman = omega

            self.cache.append({'prm': pair_removal_energy, 'prx': pair_relaxation_energy, 'orx': orbital_relaxation_energy,
                          'koo': koopman})
            
        return
 