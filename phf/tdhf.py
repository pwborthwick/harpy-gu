from __future__ import division
import phf.cit as cit
from phf.eig import solver
from mol.mol import CONSTANTS
from int.aello import aello
import int.mo_spin as mos

import numpy as np
import scipy as sp

'''
Derivation of Time Dependent Hartree-Fock (TDHF) Equations - [https://joshuagoings.com/2013/05/03/derivation-of-time-dependent-hartree-fock-tdhf-equations/]
TDHF + CIS in Python - [https://joshuagoings.com/2013/05/27/tdhf-cis-in-python/]
Numerical integrators based on the Magnus expansion for nonlinear dynamical systems 
- [https://www.researchgate.net/publication/338149703_Numerical_integrators_based_on_the_Magnus_expansion_for_nonlinear_dynamical_systems]
'''

class TDHF(object):
    #Time-dependent Hartree-Fock

    def __init__(self, scf, excitation_method='TDA'):

        self.scf = scf
        self.method = excitation_method
        self.kernel()

    def dipole(self):
        #electric dipole - for electric length gauge transition dipole

        operator      = -np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.charge_center())) 
        mos_operator  = np.array([mos.orbital_transform(self.scf, 'm+s',  operator[i]) for i in range(3)])

        return mos_operator

    def nabla(self):
        #nabla - for electric velocity gauge transition dipole

        operator       = np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'nabla', None, self.scf.mol.charge_center()))
        mos_operator   = np.array([mos.orbital_transform(self.scf, 'm+s',   operator[i]) for i in range(3)])

        return mos_operator

    def angular(self):
        #angulat momentum - for magnetic length gauge transition dipole

        operator     = np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'angular', None, self.scf.mol.charge_center()))
        mos_operator = np.array([mos.orbital_transform(self.scf, 'm+s', operator[i]) for i in range(3)])

        return mos_operator

    def kernel(self):
        #for each root calculate properties

        ci = cit.CI(self.scf)
        if self.method == 'CIS': ci.CIS() 
        if self.method == 'TDA': ci.RPA('TDA')

        #solve system
        solve = solver(roots=-1, vectors=True)
        solve.direct(ci.hamiltonian)

        #get singlets
        degeneracies = ci.degeneracies(solve.values)
        singlets = [i[0] for i in degeneracies if i[1] == 1]

        if solve.converged and (ci.transition_method == 'cis'):

            #orbital slices
            nocc, nvir = sum(self.scf.mol.nele), 2*self.scf.mol.norb - sum(self.scf.mol.nele)
            o = slice(None,nocc)
            v = slice(nocc, None)

            self.cache = []

            for root in range(solve.local_roots):

                if not solve.values[root] in singlets: continue

                #transition energies and dominant transition
                transition_energy = solve.values[root]
                energy_eV = transition_energy * CONSTANTS('hartree->eV')
                energy_nm =  CONSTANTS('eV[-1]->nm')/energy_eV

                dominant_transition = np.argmax(np.abs(solve.vectors[:, root]))
                lo, hi = (dominant_transition//nvir) + 1, (dominant_transition - nvir*(dominant_transition//nvir)) + nocc + 1

                #transition density
                tdm = ci.transition_density(solve, root).reshape(nocc, -1)

                #compute the electric length gauge transition dipole 
                length_electric = np.einsum('ia,xia->x', tdm, self.dipole()[:, o, v], optimize=True)
                
                #oscillator strength
                oscillator_length = (2/3) * transition_energy * np.einsum('p,p->', length_electric, length_electric, optimize=True)

                #compute the electric velocity gauge transition dipole 
                velocity_electric = np.einsum('ia,xia->x', tdm, self.nabla()[:, o, v], optimize=True)
                #oscillator strength
                oscillator_velocity = (2 /3) / transition_energy * np.einsum('p,p->', velocity_electric, velocity_electric, optimize=True)

                #compute the magnetic velocity gauge transition dipole
                velocity_magnetic = np.einsum('ia,xia->x', tdm, self.angular()[:, o, v], optimize=True)

                self.cache.append({'energy':[transition_energy, energy_eV, energy_nm], 'excitation':[str(lo) + '->' + str(hi), str((lo+1)//2) + '->' + str((hi+1)//2)],
                                    'electric length':[length_electric, oscillator_length], 'electric velocity':[velocity_electric, oscillator_velocity],
                                    'magnetic': velocity_magnetic})

    def format(self, roots=-1, methods='energy, electric length') :
        #output formatted tables

        cache = self.cache if roots == -1 else self.cache[:roots]

        for root, values in enumerate(cache):

            if 'energy' in methods:
                if root == 0:
                    print('\nTDHF energy analysis') 
                    print('energy:      au          eV         nm       spin   spatial\n------------------------------------------------------------')
                print(' {:2}   {:>12.6f}   {:>8.4f}   {:>8.2f}   '.format(root + 1, values['energy'][0], values['energy'][1], values['energy'][2]), end='')
                print(' {:8} {:8}'.format(values['excitation'][0], values['excitation'][1]))

        for root, values in enumerate(cache):

            if 'electric length' in methods:
                if root == 0:
                    print('\nTDHF electric dipole length gauge analysis') 
                    print('           x        y       z           S      osc.\n-----------------------------------------------------')
                print(' {:2}   {:>8.4f} {:>8.4f} {:>8.4f}   {:>8.4f}'.format(root + 1, values['electric length'][0][0], values['electric length'][0][1],
                                                                   values['electric length'][0][2], np.linalg.norm(values['electric length'][0]**2)), end='')
                print(' {:>8.4f}    '.format(values['electric length'][1]))
               
        for root, values in enumerate(cache):

            if 'electric velocity' in methods:
                if root == 0:
                    print('\nTDHF electric dipole velocity gauge analysis') 
                    print('           x        y       z           S      osc.\n-----------------------------------------------------')
                print(' {:2}   {:>8.4f} {:>8.4f} {:>8.4f}   {:>8.4f}'.format(root + 1, values['electric velocity'][0][0], values['electric velocity'][0][1],
                                                                   values['electric velocity'][0][2], np.linalg.norm(values['electric velocity'][0]**2)), end='')
                print(' {:>8.4f}    '.format(values['electric velocity'][1]))

        for root, values in enumerate(cache):

            if 'magnetic' in methods:
                if root == 0:
                    print('\nTDHF magnetic dipole velocity gauge analysis') 
                    print('           x        y       z     \n-----------------------------------------------------')
                print(' {:2}   {:>8.4f} {:>8.4f} {:>8.4f} '.format(root + 1, values['magnetic'][0], values['magnetic'][1],
                                                                   values['magnetic'][2]))

class RT_TDHF(object):

    def __init__(self, scf, pulse, dt=0.05, cycles=1000, axis='z'):

        self.scf = scf

        self.dt, self.cycles, self.axis = dt, cycles, axis

        self.pulse = pulse


    def execute(self, method='magnus'):

        #initial molecule properties
        scf_energy = self.scf.reference[0] + self.scf.reference[1]
        density, g, fock, core_h = self.scf.get('d'), self.scf.get('i'), self.scf.get('f'), self.scf.get('v') + self.scf.get('t')

        #transform ao <-> ao orthogonal
        x = sp.linalg.fractional_matrix_power(self.scf.get('s'), -0.5)
        u = sp.linalg.fractional_matrix_power(self.scf.get('s'),  0.5)
        to_orthogonal   = lambda m, s: (np.einsum('pr,rs,qs->pq', u, m, u, optimize=True) if s == 'd' else 
                                        np.einsum('rp,rs,sq->pq', x, m, x, optimize=True) )
        from_orthogonal = lambda m, s: (np.einsum('pr,rs,qs->pq', x, m, x, optimize=True) if s == 'd' else 
                                        np.einsum('rp,rs,sq->pq', u, m, u, optimize=True) )

        #initial ao orthogonal matrices
        density_p = to_orthogonal(density, 'd')
        fock_p    = to_orthogonal(fock, 'f')

        #complex time propogation step
        h = -1j * self.dt

        def orthogonalize_field(t):
            #get dipole field in ortho ao basis

            axis = ['x','y','z'].index(self.axis)

            profile = self.pulse(t) 

            mu = np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.charge_center())[axis])

            return to_orthogonal(profile * mu, 'f')

        def propogate_state(u, cycleDensity):
            #propagate time U(t) -> U(t+timeIncrement)

            density_p = np.einsum('pr,rs,qs->pq', u, cycleDensity, np.conjugate(u), optimize=True)

            #build fock in non-orthogonal ao basis
            density = from_orthogonal(density_p, 'd')
            fock = (2.0 * np.einsum('kl,imkl->im', density, g.astype('complex')) - 
                          np.einsum('kl,ikml->im', density, g.astype('complex')) +
                          core_h)


            #orthogonalize for next step
            fock_p = to_orthogonal(fock, 'f')

            return fock_p, density_p

        def second_order_magnus(h, fock_p, density_p, density):
            #second order Magnus expansion

            cache = np.empty((self.cycles, 3))

            axis = ['x','y','z'].index(self.axis)
            nmo = density_p.shape[0]
            energy = scf_energy

            for cycle in range(self.cycles):

                k = np.zeros((2,nmo,nmo)).astype('complex')

                #dipole construction with propogated density
                mu_component = np.array(aello(self.scf.mol.atom, self.scf.mol.orbital, 'dipole', None, self.scf.mol.charge_center())[axis])
                mu = -np.einsum('ii->',np.einsum('pr,rq->pq', 2.0 * density.real, mu_component, optimize=True))

                #nuclear component and charge center adjustment
                for i in self.scf.mol.atom:
                    mu += i.number * (i.center[axis] - self.scf.mol.charge_center()[axis])

                #cache values for iteration cycle
                cache[cycle, 0] = cycle * self.dt
                cache[cycle, 1] = energy
                cache[cycle, 2] = mu

                cycleDensity = density_p.copy()

                #equation (13)
                k[0] = h * (fock_p + orthogonalize_field(cycle * self.dt))
                u = sp.linalg.expm(k[0])

                fock_p, density_p = propogate_state(u, cycleDensity)

                k[1] = h * (fock_p + orthogonalize_field((cycle+1) * self.dt))
                u = sp.linalg.expm(0.5*(k[0] + k[1]))

                fock_p, density_p = propogate_state(u, cycleDensity)

                #unorthogonalise for energy calculation
                fock    = from_orthogonal(fock_p, 'f')
                density = from_orthogonal(density_p, 'd')
                energy  = np.einsum('ik,ik->', density, fock + core_h, optimize=True).real + self.scf.reference[1]
                
            return cache

        self.cache = second_order_magnus(h, fock_p, density_p, density)
