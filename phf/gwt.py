from __future__ import division
import numpy as np

from mol.mol import CONSTANTS
import int.mo_spin as mos

'''
Many-Body Perturbation Theory The GW approximation-C. Friedrich [http://www.flapw.de/pm/uploads/User-Documentation/Friedrich17.pdf]
Many-body perturbation theory software for atoms, molecules, and clusters. - Bruneval, F., Rangel, T., Hamed, S. M., Shao,
M., Yang, C., & Neaton, J. B. (2016). molgw 1:  Computer Physics Communications, 208, 149–161. [https://doi.org/10.1016/J.CPC.2016.06.019]
The GW -Method for Quantum Chemistry Applications: Theory and Implementation. - van Setten, M. J., Weigend, F., & Evers, F.
[ https://doi.org/10.1021/ct300648t]
'''
class GW(object):
    #class for simple G0W0@HF methods

    def __init__(self, scf, solve, orbital_count=None):
        #instatiate a GW object

        self.scf = scf
        self.eigensolver = solve

        self.build_mo_spin_environment()
        self.decode_range(orbital_count)

        self.shared_()

    def build_mo_spin_environment(self):
        '''
        transform orbital energies and 2-electron repulsion integrals to molecular
        spin basis
        '''

        self.eps = mos.orbital_transform(self.scf, 'm', self.scf.get('e'))
        self.g   = mos.orbital_transform(self.scf, 'm', self.scf.get('i'))
        self.f   = mos.orbital_transform(self.scf, 'm', self.scf.get('f'))

        #occupations and slices
        self.nmo, self.nocc = self.scf.mol.norb, self.scf.mol.nele[0]
        self.o, self.v, self.nvir = slice(None, self.nocc), slice(self.nocc, None), self.nmo - self.nocc

    def decode_range(self, orbital_count):
        '''
        convert string description of orbital range to numeric list
        '''

        homo, lumo = self.nocc - 1, self.nocc
        self.orbital_count = lumo if orbital_count is None else min(orbital_count, self.nmo)

    def shared_(self):
        #common values

        #short-form names
        eps, g, f = self.eps, self.g, self.f
        nocc, nvir, nmo, o, v = self.nocc, self.nvir, self.nmo, self.o, self.v

        '''
        The A and B  matrices in spin-restricted form are
        $\delta_{ij} \delta_{ab}(\epsilon_i - \epsilon_a) + 2<ai|jb> - <ji|ab>) and the B matrix is
        $2<ai|jb> - <aj|bi>). We can sub-divide these into same-spin and different-spin components
        A(s) = \delta_{ij} \delta_{ab}(\epsilon_a - \epsilon_i) + 2<ai|jb>  and  A(d) = -<ji|ab>
        B(s) = 2<ai|jb>                                                          B(d) = -<aj|bi>
        Now A+ = A(s) + A(d) = \delta_{ij} \delta_{ab}(\epsilon_a - \epsilon_i) + 2<ai|jb> - <ji|ab>
            B+ = 2<ai|jb> - <aj|bi>, we only need A+ and B+ not A- = A(s) - A(d) etc
        We form A+ + B+   and   A+ - B+ to form Hamiltonian sqrt(A-B)(A+B)sqrt(A-B) which has eigenvalues
        omega^2
        '''
        #A^+ and B^+ from molecular Hessian - ignoring exchange terms
        ones = np.ones(nmo)
        a =  np.einsum('ab,ij->iajb',np.diag(np.diag(f)[v]),np.diag(ones[o]))
        a -= np.einsum('ij,ab->iajb',np.diag(np.diag(f)[o]),np.diag(ones[v]))
        a += 2.0 * np.einsum('aijb->iajb', g[v, o, o, v], optimize=True)

        b = 2.0 * np.einsum('aijb->iajb', g[v, o, o, v], optimize=True)

        #(A^+) +  (B^+)  and  ((A^+) - (B^+)^{\frac{1}{2}} - remove negative noise for sqrt
        ab_m = a - b
        ab_m[abs(ab_m) < 1e-15] = 0.0
        ab_p, ab_m  = (a + b).reshape(nocc*nvir, nocc*nvir), np.sqrt(ab_m).reshape(nocc*nvir, nocc*nvir)

        #ensure shapes equal and construct Hermitian Hamiltonian
        assert ab_m.shape == ab_p.shape
        h = np.einsum('pr,rs,sq->pq', ab_m, ab_p, ab_m, optimize=True)

        #solve using direct solver object - eigenvalues are $\omega^2$ - density response
        self.eigensolver.direct(h)
        self.omega = np.sqrt(self.eigensolver.values.real)
        self.eigenvectors = np.einsum('ir,rs,sa->ia', ab_m, self.eigensolver.vectors,
                                                 np.diag(np.reciprocal(np.sqrt(self.omega))))


    def analytic(self, eta=1e-3):
        #G0W0 using Newton's method (analytic derivatives) and linear approximation

        self.cache = {}

        #short-form names
        eps, g, f = self.eps, self.g, self.f
        nocc, nvir, nmo, o, v = self.nocc, self.nvir, self.nmo, self.o, self.v

        #quasiparticle equation parameters
        q_nocc = min(self.orbital_count, nocc)
        q_nvir = max(self.orbital_count - nocc, 0)
        linear = True

        #get linear response solution
        omega, eigenvectors = self.omega, self.eigenvectors

        #self-energy denominantor
        d = [-eps[:nocc].reshape(-1, 1) + omega, -eps[nocc:].reshape(-1,1) - omega]

        #quasiparticle equation energies- ensemble
        qp = [eps[:(q_nocc+q_nvir)] + 1j * np.zeros(q_nocc+q_nvir)]

        #transition amplitudes
        omega_tensor = np.einsum('iapq->pqia', g[o, v, :, :], optimize=True).reshape(nmo, nmo, nocc*nvir)
        t_amp = np.sqrt(2) * np.einsum('pqi,ia->pqa', omega_tensor, eigenvectors, optimize=True)
        assert t_amp.shape == (nmo, nmo, nocc*nvir)

        #Dyson equation loop
        converged = False
        for cycle in range(self.scf.cycles):

            dia =  [d[0] + qp[0].reshape(-1,1,1) - 1j*eta, d[1] + qp[0].reshape(-1,1,1) + 1j*eta]
            assert [x.shape == (q_nocc + q_nvir, [nocc, nvir][i], len(omega)) for i, x in enumerate(dia)] == [True, True]

            #self-energy correlation contributions from holes and particles - no exchange
            sigma  = np.einsum('mip,mip,mip->m', t_amp[:(q_nocc+q_nvir), :nocc, :], np.reciprocal(dia[0]),
                                                 t_amp[:(q_nocc+q_nvir), :nocc, :], optimize=True)
            sigma += np.einsum('map,map,map->m', t_amp[:(q_nocc+q_nvir), nocc:, :], np.reciprocal(dia[1]),
                                                 t_amp[:(q_nocc+q_nvir), nocc:, :], optimize=True)

            #pole strengths - differential of sigma with respect to omega for Newton method
            z  = np.einsum('mip,mip,mip->m', t_amp[:(q_nocc+q_nvir), :nocc, :], np.reciprocal(dia[0])**2,
                                             t_amp[:(q_nocc+q_nvir), :nocc, :], optimize=True)
            z += np.einsum('map,map,map->m', t_amp[:(q_nocc+q_nvir), nocc:, :], np.reciprocal(dia[1])**2,
                                             t_amp[:(q_nocc+q_nvir), nocc:, :], optimize=True)
            z = 1.0 / (1.0 + z)

            #compute new energy
            if cycle == 0 and linear:
                self.cache['linear'] = eps[:(q_nocc+q_nvir)] + (z * sigma).real

            qp.append(eps[:(q_nocc+q_nvir)] + z * sigma)

            #check convergence of maximum difference of hole energies
            if np.max(np.abs(qp[-1] - qp[0])) < self.scf.tol:
                converged = True
                del qp[0]
                self.cache['koopman'] = eps[:(q_nocc+q_nvir)] * CONSTANTS('hartree->eV')
                self.cache['analytic'] = qp[0].real * CONSTANTS('hartree->eV')
                self.cache['sigma'] = sigma.real
                self.cache['poles'] = z.real
                if linear: self.cache['linear'] *=  CONSTANTS('hartree->eV')
                break
            else:
                del qp[0]

        else:
            print('not converged')

    def grid_(self, grid_size, grid_step):
        #construct a linear grid

        #sampling frequency grid on real line
        if (grid_size % 2 == 0): grid_size += 1
        half_width = grid_size//2
        grid = np.c_[(range(-half_width, half_width + 1))] * grid_step

        return grid, grid_size, half_width

    def numeric(self, grid_step=0.01, eta=1e-3):
        #G0W0 using 3-point stencil for derivatives

        self.cache = {}

        #short-form names
        eps, g, f = self.eps, self.g, self.f
        nocc, nvir, nmo, o, v = self.nocc, self.nvir, self.nmo, self.o, self.v

        #quasiparticle equation parameters
        q_nocc = min(self.orbital_count, nocc)
        q_nvir = max(self.orbital_count - nocc, 0)
        linear = True

        #get sampling frequency grid on real line
        grid_size = 2
        grid, grid_size, half_width = self.grid_(grid_size, grid_step)

        #ensemble grid for all energies
        omega_grid = grid + eps[:(q_nocc + q_nvir)].reshape( 1,-1)
        assert omega_grid.shape == (2 * half_width + 1, q_nocc + q_nvir)

        #get linear response solution
        omega, eigenvectors = self.omega, self.eigenvectors

        #self-energy denominantor
        d = [-eps[:nocc].reshape(-1, 1) + omega, -eps[nocc:].reshape(-1,1) - omega]

        #quasiparticle equation energies- ensemble
        qp = [eps[:(q_nocc+q_nvir)] + 1j * np.zeros(q_nocc+q_nvir)]

        #transition amplitudes
        omega_tensor = np.einsum('iapq->pqia', g[o, v, :, :], optimize=True).reshape(nmo, nmo, nocc*nvir)
        t_amp = np.sqrt(2) * np.einsum('pqi,ia->pqa', omega_tensor, eigenvectors, optimize=True)
        assert t_amp.shape == (nmo, nmo, nocc*nvir)

        dia =  [d[0] + omega_grid.reshape(grid_size,(q_nocc+q_nvir),1,1) - 1j*eta,
                d[1] + omega_grid.reshape(grid_size,(q_nocc+q_nvir),1,1) + 1j*eta]

        #self-energy correlation contributions from holes and particles on grid- no exchange
        sigma_grid  = np.einsum('nip,mnip,nip->mn', t_amp[:(q_nocc+q_nvir), :nocc, :], np.reciprocal(dia[0]),
                                                    t_amp[:(q_nocc+q_nvir), :nocc, :], optimize=True)
        sigma_grid += np.einsum('nap,mnap,nap->mn', t_amp[:(q_nocc+q_nvir), nocc:, :], np.reciprocal(dia[1]),
                                                    t_amp[:(q_nocc+q_nvir), nocc:, :], optimize=True)

        #pole strengths as derivatives using 3-point central-difference stencil
        z = np.reciprocal(1.0 - (sigma_grid[half_width+1, :] - sigma_grid[half_width-1, :]) /
                                (omega_grid[half_width+1, :] - omega_grid[half_width-1, :]))
        #check for weak self-energy pole near HF energy
        valid = [(x <= 0) or (x >=1) for x in z]
        if True in valid:
            print('weak pole near HF energy for orbitals ', [i for i, x in enumerate(valid) if x])
            z[z <= 0.0] = 0.0
            z[z >= 1.0] = 1.0

        e = eps[:(q_nocc+q_nvir)] + (z * sigma_grid[half_width, :]).real

        self.cache['koopman'] = eps[:(q_nocc+q_nvir)] * CONSTANTS('hartree->eV')
        self.cache['numeric'] = e * CONSTANTS('hartree->eV')

    def graphic(self, grid_size=600, grid_step=0.01, eta=1e-3):
        #graphical solution to G0W0

        self.cache = {}

        #short-form names
        eps, g, f = self.eps, self.g, self.f
        nocc, nvir, nmo, o, v = self.nocc, self.nvir, self.nmo, self.o, self.v

        #quasiparticle equation parameters
        q_nocc = min(self.orbital_count, nocc)
        q_nvir = max(self.orbital_count - nocc, 0)
        linear = True

        #get sampling frequency grid on real line
        grid, grid_size, half_width = self.grid_(grid_size, grid_step)

        #ensemble grid for all energies
        omega_grid = grid + eps[:(q_nocc + q_nvir)].reshape( 1,-1)
        assert omega_grid.shape == (2 * half_width + 1, q_nocc + q_nvir)

        #get linear response solution
        omega, eigenvectors = self.omega, self.eigenvectors

        #self-energy denominantor
        d = [-eps[:nocc].reshape(-1, 1) + omega, -eps[nocc:].reshape(-1,1) - omega]

        #quasiparticle equation energies- ensemble
        qp = [eps[:(q_nocc+q_nvir)] + 1j * np.zeros(q_nocc+q_nvir)]

        #transition amplitudes
        omega_tensor = np.einsum('iapq->pqia', g[o, v, :, :], optimize=True).reshape(nmo, nmo, nocc*nvir)
        t_amp = np.sqrt(2) * np.einsum('pqi,ia->pqa', omega_tensor, eigenvectors, optimize=True)
        assert t_amp.shape == (nmo, nmo, nocc*nvir)

        dia =  [d[0] + omega_grid.reshape(grid_size,(q_nocc+q_nvir),1,1) - 1j*eta,
                d[1] + omega_grid.reshape(grid_size,(q_nocc+q_nvir),1,1) + 1j*eta]

        #self-energy correlation contributions from holes and particles on grid- no exchange
        sigma_grid  = np.einsum('nip,mnip,nip->mn', t_amp[:(q_nocc+q_nvir), :nocc, :], np.reciprocal(dia[0]),
                                                    t_amp[:(q_nocc+q_nvir), :nocc, :], optimize=True)
        sigma_grid += np.einsum('nap,mnap,nap->mn', t_amp[:(q_nocc+q_nvir), nocc:, :], np.reciprocal(dia[1]),
                                                    t_amp[:(q_nocc+q_nvir), nocc:, :], optimize=True)

        import matplotlib.pyplot as plt

        max_row = 3
        height, width = max_row, max_row
        fig, axs = plt.subplots(height, width)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        for orbital_number in range(self.orbital_count):
            color = 'k' if orbital_number != (nocc-1) else 'orange'
            ax = axs.flat[orbital_number]
            ax.plot(omega_grid[:,orbital_number] * CONSTANTS('hartree->eV'),
                     sigma_grid[:,orbital_number].real * CONSTANTS('hartree->eV'), color)

            ax.set_xlabel('$\omega$ eV')
            ax.set_ylabel('$\Sigma(\omega)$ eV')
            if orbital_number <= (nocc-1): caption = 'homo-' + str(nocc - orbital_number - 1)
            if orbital_number >= nocc: caption = 'lumo+' + str(orbital_number - nocc)
            caption = caption.replace('-0','').replace('+0','')
            ax.set_title(caption, fontsize='small')
        for ax in axs.flat:
            if not bool(ax.has_data()): fig.delaxes(ax)

        fig.tight_layout()

        self.cache['plot'] = plt
