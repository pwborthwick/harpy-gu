from __future__ import division
import numpy as np
import scipy as sp

import int.mo_spin as mos
from int.aello import aello
from phf.eig import solver
from mol.mol import CONSTANTS

'''
Crawford projects 4 - [https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2304]
Chapter 6 - Modern Quantum Chemistry by A Szabo and N.S. Ostlund
C. Møller and M. S. Plesset, Phys. Rev. 46, 618 (1934) - [https://journals.aps.org/pr/abstract/10.1103/PhysRev.46.618]
LP-MP2 - [https://www.sciencedirect.com/science/article/abs/pii/000926149180078C?via%3Dihub]
'''

#*************************
#* Moller-Plesset Theory *
#*************************

class MP(object):

    def __init__(self, scf , method='mp2', parameter=None):

        self.method = method
        self.scf = scf

        #despatcher 
        if self.method == 'MP2':     self.mp(2)       
        if self.method == 'MP3':     self.mp(3)
        if self.method == 'MP2r':    self.mp2r()
        if self.method == 'MP2rdm1': 
            parameter = 'unrelaxed' if parameter == None else 'relaxed'
            self.mp2rdm1(mode=parameter)

        if self.method == 'MP2mu':   self.mp2mu()
        if self.method == 'OMP2':    self.mp2oo()

        if self.method == 'LT-MP2': 
            parameter = 100 if parameter == None else parameter
            self.mp2ltr(grid=parameter)

        if self.method == 'MP2rno':  self.mp2rno()

    def mp(self, level=2):

        eps = mos.orbital_deltas(self.scf, 2)                                  #returns a list of orbital differences upto and including level

        eps_denominator = np.reciprocal(eps[1])                                #denominator at level 2 

        g = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))          #spin 2-electron repulsion integrals

        nocc = sum(self.scf.mol.nele)                                          #occupied spin orbitals in number of alpha + beta electrons
        o, v = slice(None, nocc), slice(nocc, None)                            #orbital occupation slices

        td_amplitude = g[o, o, v, v]*eps_denominator                           #cluster doubles amplitude

        if level == 2:

            mp2_energy = 0.25 * np.einsum('ijab,ijab->', g[o, o, v, v], td_amplitude, optimize=True)

            self.correction = mp2_energy

        if level == 3:

            mp2_energy =  pow(1/2, 3) * np.einsum('ijab,klij,abkl,klab', td_amplitude, g[o,o,o,o], g[v,v,o,o], eps_denominator)
            mp2_energy += pow(1/2, 0) * np.einsum('ijab,akic,bcjk,jkbc', td_amplitude, g[v,o,o,v], g[v,v,o,o], eps_denominator)
            mp2_energy += pow(1/2, 3) * np.einsum('ijab,abcd,cdij,ijcd', td_amplitude, g[v,v,v,v], g[v,v,o,o], eps_denominator)
    
            self.correction = mp2_energy


    def mp2r(self):
        #spin-restricted MP2 versions (parallel-antiparallel, spin-component scaled)

        eps = mos.orbital_deltas(self.scf, 2, mo='x')                          #returns a list of orbital differences upto and including level

        eps_denominator = np.reciprocal(eps[1])                                #denominator at level 2 

        g = mos.orbital_transform(self.scf, 'm', self.scf.get('i'))            #get spatial orbitals in molecular basis

        nocc = sum(self.scf.mol.nele)//2                                       #occupied spin orbitals in number of alpha + beta electrons
        o, v = slice(None, nocc), slice(nocc, None)                            #orbital occupation slices

        g = g[o, v, o, v]
        parallel      = np.einsum('iajb,iajb,ijab->', g, g, eps_denominator) - np.einsum('iajb,ibja,ijab->', g, g, eps_denominator)
        anti_parallel = np.einsum('iajb,iajb,ijab->', g, g, eps_denominator)

        #append spin-component scaled contribution
        self.correction = [parallel, anti_parallel,  parallel/3.0 + 1.2 * anti_parallel]

    def mp2rdm1(self, mode='unrelaxed'):
        #MP2 unrelxed density matrix

        eps = mos.orbital_deltas(self.scf, 2)                                  #returns a list of orbital differences upto and including level

        eps_denominator = np.reciprocal(eps[1])                                #denominator at level 2 

        g = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))          #get spatial orbitals in molecular basis

        nocc = sum(self.scf.mol.nele)                                          #occupied spin orbitals in number of alpha + beta electrons
        o, v = slice(None, nocc), slice(nocc, None)                            #orbital occupation slices

        td = g[o, o, v, v]*eps_denominator                                     #cluster doubles amplitude

        oo = -0.5* np.einsum('ikab,jkab->ij', td, td, optimize=True)           #particle block and enforce symmetry
        oo = 0.5 * (oo + np.transpose(oo))

        vv = 0.5 * np.einsum('ijac,ijbc->ab', td, td, optimize=True)           #hole block and enforce symmetry
        vv = 0.5 * (vv + np.transpose(vv))

        ov = -0.5 * (np.einsum('ijbc,jabc->ia', td, g[o,v,v,v], optimize=True) +
                     np.einsum('jkib,jkab->ia', g[o,o,o,v], td, optimize=True) ) * np.reciprocal(eps[0])

        oo += np.eye(nocc)                                                     #add reference density
        if mode == 'unrelaxed': ov = np.zeros_like(ov)                         #zero ov block for unrelaxed

        dm = np.block([[oo            , ov], 
                       [ov.transpose(), vv]])

        self.mp2dm = dm

    def mp2mu(self):
        #the MP2 unrelaxed dipole

        debyes = CONSTANTS('au->debye')
                                                                               #get dipole components and work in AO
        dipole = np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.gauge)) 

        c = self.scf.get('c')
        mu_mo_electric = np.einsum('rp,xrs,sq->xpq', c, dipole, c, optimize=True)
                               
        mu_nuclear = np.zeros(3)                                               #nuclear component and charge center adjustment
        for i in range(3):
            for j in range(self.scf.mol.natm):
                mu_nuclear[i] += self.scf.mol.atom[j].number * (self.scf.mol.atom[j].center[i] - self.scf.mol.gauge[i])

        #unrelaxed MP2 density
        self.mp2rdm1()                                                         #get mp2 density and comvert to spatial (AO)
        dm = mos.spin_to_spatial(self.mp2dm)
        mu_mp2_electric = np.einsum('xij,ij->x', mu_mo_electric, dm, optimize=True)
        mu_mp2_unrelaxed = list((mu_nuclear - mu_mp2_electric) * debyes)

        #relaxed MP2 density
        self.mp2rdm1(mode='relaxed')
        dm = mos.spin_to_spatial(self.mp2dm)
        mu_mp2_electric = np.einsum('xij,ij->x', mu_mo_electric, dm, optimize=True)
        mu_mp2_relaxed = list((mu_nuclear - mu_mp2_electric) * debyes)

        #reference HF dipole
        mu = (-np.einsum('xij,ij->x', dipole, 2.0 * self.scf.get('d'), optimize=True) + mu_nuclear) * debyes

        self.dipole = [mu, mu_mp2_unrelaxed, mu_mp2_relaxed]

    def mp2oo(self):
        #orbital-opimized MP2

        scf = self.scf                                                         #short form

        nuclear_repulsion = scf.mol.nuclear_repulsion()                        #nuclear repulsion energy

        nocc = sum(scf.mol.nele)                                               #orbital occupations
        nsbf = scf.mol.norb * 2                                     
        nvir =  nsbf - nocc

        h = scf.get('t') + scf.get('v')                                        #ao quantities

        gao = mos.orbital_transform(scf, 's', scf.get('i'))                    #ao-spin basis quantities
        hao = mos.orbital_transform(scf, 's', h)
        eps = mos.orbital_transform(scf, 's', scf.get('e'))

        c = mos.orbital_transform(scf, 'b', scf.get('c'))                      #ao to mo block transformation tensor

        gmo = mos.orbital_transform(scf, 'm+s', scf.get('i'))                  #mo spin basis quantities
        hmo = mos.orbital_transform(scf, 'm+s', h)

        o, v, n = slice(None, nocc), slice(nocc, None), np.newaxis             #orbital slices

        amplitude = np.zeros((nocc, nocc, nvir, nvir))                         #tensor initialisation 
        opd = np.zeros_like(hmo) 
        x = np.zeros_like(hmo)      

        dm = np.zeros_like(opd)                                                #HF reference density matrix
        dm[o, o] = np.eye(nocc)
                                                                               #get HF energy
        fock = mos.orbital_transform(scf, 'm+s', scf.get('f'))
        scf_energy =  np.einsum('ii', fock[o, o]) - 0.5 * np.einsum('ijij', gmo[o, o, o, o])

        last_cycle_energy = 0.0                                                #iteration control
        extra_cycle = False
        if not scf.mol.silent: print(' cycle           energy                \u0394E')

        for cycle in range(1, scf.cycles):

            f = hmo + np.einsum('piqi -> pq', gmo[:, o, :, o], optimize=True)  #build the Fock matrix

            fprime = f.copy()                                                  #get diagonal and zero-traced Fock
            np.fill_diagonal(fprime, 0)
            eps = f.diagonal()

            orbital_difference = mos.orbital_deltas(scf, 2, 's', e=eps)        #compute the orbital energy denominators

            td = gmo[o, o, v, v]                                               #update mp2 t2 amplitudes
            t  = np.einsum('ac,ijcb -> ijab', fprime[v, v], amplitude, optimize=True)
            td += t - t.transpose(0, 1, 3, 2)
            t = -np.einsum('ki,kjab -> ijab', fprime[o, o], amplitude, optimize=True)
            td += t - t.transpose(1, 0, 2, 3)
            
            amplitude = td * np.reciprocal(orbital_difference[1])
                                                                               #construct one particle density matrix
            opd[v, v] =  0.5 * np.einsum('ijac,ijbc -> ba', amplitude, amplitude, optimize=True)
            opd[o, o] = -0.5 * np.einsum('abjk,ikab -> ji', amplitude.T, amplitude, optimize=True)
                                                                               #construct two particle density matrix
            t = np.einsum('rp,sq -> rspq', opd, dm, optimize=True)
            tpdm = t - t.transpose(1, 0, 2, 3) - t.transpose(0, 1, 3, 2) + t.transpose(1, 0, 3, 2)
            t = np.einsum('rp,sq->rspq', dm, dm, optimize=True)
            tpdm += t - t.transpose(1, 0, 2, 3) 
            tpdm[o, o, v, v], tpdm[v, v, o, o] = amplitude, amplitude.transpose()

            opd  += dm                                                         #add in one particle reference density
                                                                               #Newton-Raphson step
            fock = np.einsum('pr,rq->pq', hmo, opd) + 0.5 * np.einsum('prst,stqr -> pq', gmo, tpdm, optimize=True)
            x[o, v] = ((fock - fock.T)[o, v]) * np.reciprocal(orbital_difference[0])
            u = sp.linalg.expm(x - x.transpose())                              #Newton-Raphson orbital rotation matrix

            c = np.einsum('pr,rq->pq', c, u, optimize=True)                    #rotate spin-orbital coefficients

            hmo = mos.orbital_transform(scf, 'm', hao, c)                      #transform electron integrals with rotated mo's
            gmo = mos.orbital_transform(scf, 'm', gao, c)
                                                                               #compute the energy
            oomp2_energy = (np.einsum('pq,qp->', hmo, opd, optimize=True) + 
                            0.25 * np.einsum('pqrs,rspq->', gmo, tpdm, optimize=True))

            if not scf.mol.silent:
                print('  {:2}       {:>15.10f}     {:>15.10f}'.format(cycle, oomp2_energy, (oomp2_energy-last_cycle_energy)))

            if (abs(oomp2_energy - last_cycle_energy)) < scf.tol:              #check for convergence
                if extra_cycle: break
                extra_cycle = not extra_cycle

            last_cycle_energy = oomp2_energy                                   #update last cycle energy

            self.correction = oomp2_energy - scf_energy

    def mp2ltr(self, grid=100):
        #Laplace Transform restricted-spin MP2

        scf = self.scf                                                         #short-cut                                         

        nmo = scf.mol.norb                                                     #metrics and orbital slices
        nocc = sum(scf.mol.nele)//2
        nvir = nmo - nocc
        
        o, v, V, n = slice(None, nocc), slice(nocc, None), slice(None, nvir), np.newaxis

        c   = scf.get('c')                                                     #ao quantities
        i   = scf.get('i')
        eps = scf.get('e')

        g = np.einsum('pi,pqrs->iqrs', c[:,o], i, optimize=True)[o, :, :, :]   #2-electron integrals sliced in mo basis
        g = np.einsum('qa,pqrs->pars', c[:,v], g, optimize=True)[:, V, :, :]
        g = np.einsum('rj,pqrs->pqjs', c[:,o], g, optimize=True)[:, :, o, :]
        g = np.einsum('sb,pqrs->pqrb', c[:,v], g, optimize=True)[:, :, :, V]
     
        mesh, weights = np.polynomial.laguerre.laggauss(grid)                  #Gauss-Leguerre quadrature
        weights *= np.exp(mesh)

        lt_energy = [0.0, 0.0, 0.0]

        for cycle in range(grid):                                              #cycle over grid points

                                                                               #compute amplitudes
            amp_ov = np.einsum('i,a->ia', np.exp( mesh[cycle] * eps[o]), np.exp(-mesh[cycle] * eps[v]), optimize=True)
            amplitude = np.einsum('ia,jb,iajb->iajb', amp_ov, amp_ov, g, optimize=True)
      
            mplt_energy = [0.0, 0.0]                                           #Laplace mp2 energies
            mplt_energy[1] = np.einsum('iajb,iajb->', amplitude, g, optimize=True)  
            mplt_energy[0] = mplt_energy[1] - np.einsum('iajb,ibja->', amplitude, g, optimize=True)

            for e in range(0, 2):                                              #apply integration weights
                lt_energy[e] -= mplt_energy[e] * weights[cycle]
            lt_energy[2] = sum(lt_energy[:2])
        
        self.correction = lt_energy

    def mp2rno(self):
        #MP2 natural orbital in spin-restriced basis

        eps = mos.orbital_deltas(self.scf, 2, mo='x')                          #returns a list of orbital differences upto and including level

        eps_denominator = np.reciprocal(eps[1])                                #denominator at level 2 

        g = mos.orbital_transform(self.scf, 'm', self.scf.get('i'))            #get spatial orbitals in molecular basis

        nocc = sum(self.scf.mol.nele)//2                                       #occupied spin orbitals in number of alpha + beta electrons
        o, v = slice(None, nocc), slice(nocc, None)                            #orbital occupation slices

        g = g[o, v, o, v]                                                      #compute MP2 factors
        ga = 2.0 * g - g.transpose( 0, 3, 2, 1)
        gb = g * eps_denominator.transpose(0, 2, 1, 3)

        co = np.einsum('iakb,jakb->ij', ga, gb, optimize=True)
        cv = np.einsum('iajb,icjb->ca', ga, gb, optimize=True)
 
        dm = np.zeros_like(self.scf.get('s'))                                  #compute density

        dm[o, o] += 0.25 * (co + co.transpose()) + 2.0 * np.diag(np.ones(nocc))#occupied block and reference density
        dm[v, v] -= 0.25 * (cv + cv.transpose())                               #virtual block

        solve = solver(roots=-1, vectors=True)
        solve.direct(dm)                                                       #diagonalize

        self.natural_mo = [solve.values, solve.vectors]
