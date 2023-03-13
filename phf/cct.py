from __future__ import division
import numpy as np

import int.mo_spin as mos
from scf.diis import dii_subspace
from phf.mpt import MP

'''
CCSD - https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2305
CCSD(T) - https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306
CC Theory - MANY-BODY METHODS IN CHEMISTRY AND PHYSICS MBPT and Coupled-Cluster Theory Shavitt & Bartlett
CCD & CCSD https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html/CCM.html
CCSDT-n - https://www.duo.uio.no/bitstream/handle/10852/42315/1/OtnorliMasterThesis.pdf
all methods - https://github.com/pwborthwick/harpy/blob/main/document/Coupled_Cluster.ipynb
restricted spin CCD and CCSD - Hirata, S.; Podeszwa, R.; Tobita, M.; Bartlett, R. J. J. Chem. Phys. 2004, 120 (6), 2581
'''

class CC(object):
    #class for Coupled-Cluster methods

    def __init__(self, scf, cycles=50, tol=1e-10, diis=True):
        #initialise CC object

        self.scf = scf
        self.cycles, self.tol, self.diis = cycles, tol, diis

        self.nocc = sum(self.scf.mol.nele)
        self.o, self.v = slice(None, self.nocc), slice(self.nocc, None)

        self.lambda_ccsd, self.code = False, None

    def method(self, code='ccd', ts=None, td=None, silent=True):
        #execute the method implied by 'code'

        if not code in ['ccd', 'ccsd', 'ccsd(t)', 'ccsdt-1a', 'ccsdt-1b',
                        'lccd', 'lccsd', 'qcisd']:
            print('coupled-cluster code [', code, '] not recognised')
            self.cache = None
            return

        #compute molecular spin Fock and 2-electron repulsion
        self.fs = mos.orbital_transform(self.scf, 'm+s', self.scf.get('f'))
        self.gs = mos.orbital_transform(self.scf, 'm+s', self.scf.get('i'))

        o, v = self.o, self.v

        #initial amplitudes
        self.deltas = mos.orbital_deltas(self.scf, 3)
        self.ts = ts if (isinstance(ts, np.ndarray) and ts.shape[0] == self.nocc) else np.zeros_like(self.deltas[0])
        self.td = td if (isinstance(td, np.ndarray) and td.shape[0] == self.nocc) else self.gs[o, o, v, v] * np.reciprocal(self.deltas[1])
        self.tt = np.zeros_like(self.deltas[2])

        self.code = code
        self.is_linear   = (self.code in ['lccd', 'lccsd'])
        self.has_triples = (self.code in ['ccsdt-1a', 'ccsdt-1b'])

        func_method = self.update_amplitudes if not self.is_linear else self.update_linear_amplitudes

        self.iterator(func_method, silent)

        return

    def tau(self, tilde=True):
        #tau expressions

        tau = self.td.copy()
        factor = 0.5 if tilde else 1.0
        t = factor * np.einsum('ia,jb->ijab', self.ts, self.ts, optimize=True)

        tau += t - t.transpose(0, 1, 3, 2)

        return tau

    def intermediates(self, code, tilde=True):
        #Coupled-cluster intermediate expressions

        o, v = self.o, self.v

        if tilde:

            if not code in ['oo', 'vv', 'ov', 'oooo', 'vvvv', 'ovvo']:
                print('no pre-computed slice [', code, '] - use transpose')
                exit()

            if code == 'oo':
                im = self.fs[o, o].copy()
                np.fill_diagonal(im, 0.0)
                im += 0.5 * np.einsum('ie,me->mi', self.ts, self.fs[o, v], optimize=True)
                im += np.einsum('ne,mnie->mi', self.ts, self.gs[o, o, o, v], optimize=True)
                im += 0.5 * np.einsum('inef,mnef->mi', self.tau(), self.gs[o, o, v, v], optimize=True)

            if code == 'vv':
                im = self.fs[v, v].copy()
                np.fill_diagonal(im, 0.0)
                im -= 0.5 * np.einsum('ma,me->ae', self.ts, self.fs[o, v],optimize=True)
                im += np.einsum('mf,mafe->ae', self.ts, self.gs[o, v, v, v], optimize=True)
                im -= 0.5 * np.einsum('mnaf,mnef->ae', self.tau(), self.gs[o, o, v, v], optimize=True)

            if code == 'ov':
                im = self.fs[o,v].copy()
                im += np.einsum('nf,mnef->me', self.ts, self.gs[o, o, v, v], optimize=True)

            if code == 'oooo':
                im = self.gs[o, o, o, o].copy()
                t = np.einsum('je,mnie->mnij', self.ts, self.gs[o, o, o, v], optimize=True)
                im += t - t.transpose(0,1,3,2)
                im += 0.25 * np.einsum('ijef,mnef->mnij', self.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if code == 'vvvv':
                im = self.gs[v, v, v, v].copy()
                t = -np.einsum('mb,amef->abef', self.ts, self.gs[v, o, v, v], optimize=True)
                im += (t - t.transpose(1,0,2,3))
                im += 0.25 * np.einsum('mnab,mnef->abef', self.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if code == 'ovvo':
                im = self.gs[o, v, v, o].copy()
                im += np.einsum('jf,mbef->mbej', self.ts, self.gs[o, v, v, v], optimize=True)
                im -= np.einsum('nb,mnej->mbej', self.ts, self.gs[o, o, v, o], optimize=True)
                im -= 0.5 * np.einsum('jnfb,mnef->mbej', self.td, self.gs[o, o, v, v], optimize=True)
                im -= np.einsum('jf,nb,mnef->mbej', self.ts, self.ts, self.gs[o, o, v, v], optimize=True)

        if not tilde:

            if not code in ['oo','vv','ov','oooo','vvvv','ovvo','ooov','vovv','ovoo','vvvo','OO','VV']:
                print('no pre-evaluated slice [', code, '] - transpose axes')
                exit()

            if code == 'vv':
                im = self.intermediates('vv')
                im -= 0.5 * np.einsum('me,ma->ae', self.intermediates('ov'), self.ts, optimize=True)

            if code == 'oo':
                im = self.intermediates('oo')
                im += 0.5 * np.einsum('me,ie->mi', self.intermediates('ov'), self.ts, optimize=True)

            if code == 'ov':
                im = self.intermediates('ov')

            if code == 'oooo':
                im  = self.intermediates('oooo')
                im += 0.25 * np.einsum('ijef,mnef->mnij', self.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if code == 'vvvv':
                im = self.intermediates('vvvv')
                im += 0.25 * np.einsum('mnab,mnef->abef', self.tau(tilde=False), self.gs[o, o, v, v], optimize=True)

            if code == 'ovvo':
                im = self.intermediates('ovvo')
                im -= 0.5 * np.einsum('jnfb,mnef->mbej', self.td, self.gs[o, o, v, v], optimize=True)

            if code == 'ooov':
                im =  self.gs[o, o, o, v].copy()
                im += np.einsum('if,mnfe->mnie', self.ts, self.gs[o, o, v, v], optimize=True)

            if code == 'vovv':
                im = self.gs[v, o, v, v].copy()
                im += np.einsum('na,mnef->amef', self.ts, self.gs[o, o, v, v], optimize=True)

            if code == 'ovoo':
                im = self.gs[o, v, o, o].copy()
                im -= np.einsum('me,ijbe->mbij', self.intermediates('ov'), self.td, optimize=True)
                im -= np.einsum('nb,mnij->mbij', self.ts, self.intermediates('oooo', tilde=False), optimize=True)
                im += 0.5 * np.einsum('ijef,mbef->mbij', self.tau(tilde=False), self.gs[o, v, v, v], optimize=True)
                t = np.einsum('jnbe,mnie->mbij', self.td, self.gs[o, o, o, v], optimize=True)
                im += t - t.transpose(0,1,3,2)
                t = np.einsum('ie,mbej->mbij', self.ts, self.gs[o, v, v, o], optimize=True)
                t -= np.einsum('ie,njbf,mnef->mbij', self.ts, self.td, self.gs[o, o, v, v], optimize=True)
                im += t - t.transpose(0,1,3,2)

            if code == 'vvvo':
                im = self.gs[v, v, v, o].copy()
                im -= np.einsum('me,miab->abei', self.intermediates('ov'), self.td, optimize=True)
                im += np.einsum('if,abef->abei', self.ts, self.intermediates('vvvv', tilde=False), optimize=True)
                im += 0.5 * np.einsum('mnab,mnei->abei', self.tau(tilde=False), self.gs[o, o, v, o], optimize=True)
                t = -np.einsum('miaf,mbef->abei', self.td, self.gs[o, v, v, v], optimize=True)
                im += t - t.transpose(1,0,2,3)
                t = -np.einsum('ma,mbei->abei', self.ts, self.gs[o, v, v, o], optimize=True)
                t += np.einsum('ma,nibf,mnef->abei', self.ts, self.td, self.gs[o, o, v, v], optimize=True)
                im += t - t.transpose(1,0,2,3)

            if code == 'VV':
                im = -0.5 * np.einsum('afmn,mnef->ae', self.ld, self.td, optimize=True)

            if code == 'OO':
                im =  0.5 * np.einsum('efin,mnef->mi', self.ld, self.td, optimize=True)

        return im

    def update_qcisd_amplitudes(self):
        #quadratic configuration interaction singles and doubles

            o, v = self.o, self.v
            Ioo, Ivv = self.fs[o, o], self.fs[v, v]
            np.fill_diagonal(Ioo, 0.0) ; np.fill_diagonal(Ivv, 0.0)

            t1 = np.einsum('ic,ac->ia', self.ts, Ivv, optimize=True)
            t1 -= np.einsum('ka,ki->ia', self.ts, Ioo, optimize=True)
            t1 += np.einsum('kc,akic->ia', self.ts, self.gs[v, o, o, v], optimize=True)
            t1 += np.einsum('ikac,kc->ia', self.td, self.fs[o,v], optimize=True)
            t1 += 0.5 * np.einsum('kicd,kacd->ia', self.td, self.gs[o, v, v, v], optimize=True)
            t1 -= 0.5 * np.einsum('klca,klci->ia', self.td, self.gs[o, o, v, o], optimize=True)
            t1 -= 0.5 * np.einsum('ic,klda,lkcd->ia', self.ts, self.td, self.gs[o, o, v, v], optimize=True)
            t1 -= 0.5 * np.einsum('ka,licd,lkcd->ia', self.ts, self.td, self.gs[o, o, v, v], optimize=True)
            t1 += np.einsum('kc,lida,klcd->ia', self.ts, self.td, self.gs[o, o, v, v], optimize=True)

            t2 = self.gs[o, o, v, v].copy()
            t = np.einsum('ic,abcj->ijab', self.ts, self.gs[v, v, v, o], optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -np.einsum('ka,kbij->ijab', self.ts, self.gs[o, v, o, o], optimize=True)
            t2 += t - t.transpose(0,1,3,2)
            t = np.einsum('ijac,bc->ijab', self.td, Ivv, optimize=True)
            t2 += t - t.transpose(0,1,3,2)
            t = -np.einsum('ikab,kj->ijab', self.td, Ioo, optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t2 += 0.5 * np.einsum('ijcd,abcd->ijab', self.td, self.gs[v , v, v, v], optimize=True)
            t2 += 0.5 * np.einsum('klab,klij->ijab', self.td, self.gs[o, o ,o ,o], optimize=True)
            t = np.einsum('ikac,kbcj->ijab', self.td, self.gs[o , v, v, o])
            t2 += t - t.transpose(0,1,3,2)  - t.transpose(1,0,2,3) + t.transpose(1,0,3,2)
            t2 += 0.25 * np.einsum('ijcd,klab,klcd->ijab', self.td, self.td, self.gs[o, o, v, v], optimize=True)
            t = np.einsum('ikac,jlbd,klcd->ijab', self.td, self.td, self.gs[o, o, v, v], optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -0.5 * np.einsum('ikdc,ljab,klcd->ijab', self.td, self.td, self.gs[o, o, v, v], optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -0.5 * np.einsum('lkac,ijdb,klcd->ijab', self.td, self.td, self.gs[o, o, v, v], optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            return t1, t2

    def update_triples_amplitudes(self, t1, t2):
        #add terms for various approximate triples

        o, v = self.o, self.v

        if self.code in ['ccsdt-1a', 'ccsdt-1b']:
            t1 += 0.25 * np.einsum('mnef,imnaef->ia', self.gs[o, o, v, v], self.tt, optimize=True)

            t2 +=  np.einsum('em,ijmabe->ijab', self.fs[v, o], self.tt, optimize=True)
            t =  0.5 * np.einsum('ijmaef,bmef->ijab', self.tt, self.gs[v, o, v, v], optimize=True)
            t2 += t - t.transpose(0,1,3,2)
            t = -0.5 * np.einsum('imnabe,mnje->ijab', self.tt, self.gs[o, o, o, v], optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t = np.einsum('ijae,bcek->ijkabc', self.td, self.gs[v, v, v, o], optimize=True)
            t = t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
            t3 = t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
            t = -np.einsum('imab,mcjk->ijkabc', self.td, self.gs[o, v, o, o], optimize=True)
            t =  t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
            t3 += t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
            t = np.einsum('ce,ijkabe->ijkabc', self.fs[v, v], self.tt, optimize=True)
            t3 += t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
            t = -np.einsum('km,ijmabc->ijkabc', self.fs[o, o], self.tt, optimize=True)
            t3 += t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)

        if self.code in ['ccsdt-1b']:
            t2 += np.einsum('mnef,me,nijfab->ijab', self.gs[o, o, v, v], self.ts, self.tt, optimize=True)
            t = -0.5 * np.einsum('mnef,ma,injefb->ijab', self.gs[o, o, v, v], self.ts, self.tt, optimize=True)
            t2 += t - t.transpose(0,1,3,2)
            t = -0.5 * np.einsum('mnef,ie,mnjafb->ijab', self.gs[o, o, v, v], self.ts, self.tt, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

        return t1, t2, t3

    def update_linear_amplitudes(self):
        #linear amplitudes next cycle

        o, v = self.o, self.v

        #doubles amplitdes - no singles for linear
        Ioo, Ivv = self.fs[o,o], self.fs[v,v]
        np.fill_diagonal(Ioo, 0.0) ; np.fill_diagonal(Ivv, 0.0)

        t2 = self.gs[o, o, v, v].copy()

        t = np.einsum('be,ijae->ijab', Ivv, self.td, optimize=True)
        t2 += t - t.transpose(0,1,3,2)
        t = -np.einsum('mj,imab->ijab', Ioo, self.td, optimize=True)
        t2 += t - t.transpose(1,0,2,3)

        t2 += 0.5 * np.einsum('abef,ijef->ijab', self.gs[v, v, v, v], self.td, optimize=True)
        t2 += 0.5 * np.einsum('mnij,mnab->ijab', self.gs[o, o, o, o], self.td, optimize=True)

        t = np.einsum('mbej,imae->ijab', self.gs[o, v, v, o], self.td, optimize=True)
        t2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

        if self.code == 'lccsd':

            t1 = self.fs[o,v].copy()
            t1 += np.einsum('me,imae->ia', self.fs[o,v], self.td, optimize=True)
            t1 += 0.5 * np.einsum('efam,imef->ia', self.gs[v, v, v, o], self.td, optimize=True)
            t1 -= 0.5 * np.einsum('iemn,mnae->ia', self.gs[o, v, o, o], self.td, optimize=True)
            t1 += np.einsum('ae,ie->ia', Ivv, self.ts, optimize=True)
            t1 -= np.einsum('mi,ma->ia', Ioo, self.ts, optimize=True)
            t1 += np.einsum('ieam,me->ia', self.gs[o, v, v, o], self.ts, optimize=True)

            t = np.einsum('ejab,ie->ijab', self.gs[v, o, v, v], self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3)
            t = -np.einsum('ijmb,ma->ijab', self.gs[o, o, o, v], self.ts, optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            self.ts = t1 * np.reciprocal(self.deltas[0])
        self.td = t2 * np.reciprocal(self.deltas[1])


    def update_amplitudes(self):
        #compute the next cycle amplitudes

        o, v = self.o, self.v

        Ioo, Ivv, Iov = self.intermediates('oo'), self.intermediates('vv'), self.intermediates('ov')

        if self.code == 'ccd':
            t1 = self.ts.copy()
        elif self.code in ['ccsd', 'ccsd(t)', 'ccsdt-1a', 'ccsdt-1b']:
            t1 = self.fs[o, v].copy()
            t1 += np.einsum('ie,ae->ia', self.ts, Ivv, optimize=True)
            t1 -= np.einsum('ma,mi->ia', self.ts, Ioo, optimize=True)
            t1 += np.einsum('imae,me->ia', self.td, Iov, optimize=True)
            t1 -= np.einsum('nf,naif->ia', self.ts, self.gs[o, v, o, v], optimize=True)
            t1 -= 0.5 * np.einsum('imef,maef->ia', self.td, self.gs[o, v, v, v], optimize=True)
            t1 -= 0.5 * np.einsum('mnae,nmei->ia', self.td, self.gs[o, o, v, o], optimize=True)

        t2 = self.gs[o, o, v, v].copy()

        if self.code in ['ccd', 'ccsd', 'ccsd(t)', 'ccsdt-1a', 'ccsdt-1b']:
            Ioooo, Ivvvv, Iovvo = self.intermediates('oooo'), self.intermediates('vvvv'), self.intermediates('ovvo')

            t2 += 0.5 * np.einsum('mnab,mnij->ijab', self.tau(tilde=False), Ioooo, optimize=True)

            t2 += 0.5 * np.einsum('ijef,abef->ijab', self.tau(tilde=False), Ivvvv, optimize=True)

            t = np.einsum('abej,ie->ijab', self.gs[v, v, v, o], self.ts, optimize=True)
            t2 += t - t.transpose(1,0,2,3)

            t = np.einsum('ma,mbij->ijab', self.ts, self.gs[o, v, o, o], optimize=True)
            t2 += -(t - t.transpose(0,1,3,2))

            t = np.einsum('imae,mbej->ijab', self.td, Iovvo, optimize=True)
            t -= np.einsum('ie,ma,mbej->ijab', self.ts, self.ts, self.gs[o, v, v, o], optimize=True)
            t2 += t - t.transpose(0,1,3,2) - t.transpose(1,0,2,3) + t.transpose(1,0,3,2)

            t = np.einsum('ijae,be->ijab', self.td, Ivv, optimize=True)
            t -= 0.5 * np.einsum('ijae,mb,me->ijab', self.td, self.ts, Iov, optimize=True)
            t2 += t - t.transpose(0,1,3,2)

            t = np.einsum('imab,mj->ijab', self.td, Ioo, optimize=True)
            t += 0.5 * np.einsum('imab,je,me->ijab', self.td, self.ts, Iov, optimize=True)
            t2 += -(t - t.transpose(1,0,2,3))

        if self.has_triples:
            t1, t2, t3 = self.update_triples_amplitudes(t1, t2)
            self.tt = t3 * np.reciprocal(self.deltas[2]) + self.tt

        if self.code == 'qcisd':
            t1, t2 = self.update_qcisd_amplitudes()

        #update amplitudes
        self.ts = t1 * np.reciprocal(self.deltas[0])
        self.td = t2 * np.reciprocal(self.deltas[1])

    def cluster_energy(self):
        #compute the coupled-cluster energy correction

        o, v = self.o, self.v

        energy_linear = np.einsum('ia,ia->', self.fs[o, v], self.ts, optimize=True)

        energy_linear += 0.25 * np.einsum('ijab,ijab->', self.gs[o, o, v, v], self.td, optimize=True)

        #non-linear term
        energy_quadratic = 0.0 if self.code in ['lccd', 'lccsd', 'qcisd'] else 0.5 * np.einsum('ijab,ia,jb->', self.gs[o, o, v, v], self.ts, self.ts, optimize=True)

        return energy_linear + energy_quadratic

    def perturbative_triples(self):
        #compute the perturbative triples contribution

        o, v = self.o, self.v

        #cyclic permutation list
        cyclic = {'ijkbac':-1,'ijkcba':-1,'jikabc':-1,'jikbac':+1,
                  'jikcba':+1,'kjiabc':-1,'kjibac':+1,'kjicba':+1}

        #disconnected triples amplitudes
        t = np.einsum('ia,jkbc->ijkabc', self.ts, self.gs[o, o, v, v], optimize=True)
        disconnected_triples = t.copy()
        for i in cyclic:
            disconnected_triples += cyclic[i] * np.einsum('ijkabc->' + i, t, optimize=True)

        #connected triples amplitudes
        t =  np.einsum('jkae,eibc->ijkabc', self.td, self.gs[v, o, v, v], optimize=True)
        t -= np.einsum('imbc,majk->ijkabc', self.td, self.gs[o, v, o, o], optimize=True)
        connected_triples = t.copy()
        for i in cyclic:
            connected_triples += cyclic[i] * np.einsum('ijkabc->' + i, t, optimize=True)

        triples = (disconnected_triples + connected_triples) * self.deltas[2]

        return np.einsum('ijkabc,ijkabc->', connected_triples, triples) / 36.0

    def iterator(self, func, silent):
        #consistent field coupled-cluster iterations

        #initialise diis buffers
        if self.diis:
            diis = dii_subspace(self.scf.diis_size, type='c')
            diis.cache = [[self.ts, self.td, self.tt]]

        cycle_energy = [self.cluster_energy()]

        for cycle in range(self.cycles):

            #store pre-update amplitudes
            if self.diis:
                diis.append([self.ts, self.td, self.tt])

            func()

            #calculate current cycle energy
            cycle_energy.append(self.cluster_energy())

            #test convergence
            delta_energy = np.abs(cycle_energy[-2] - cycle_energy[-1])
            if delta_energy < self.tol:
                self.converged = True
                if self.code != 'ccsd(t)':
                    self.cache = [cycle_energy[-1]]
                else:
                    self.cache = [cycle_energy[-1], self.perturbative_triples()]
                return
            else:
                if not silent: print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f}   |diis| = {:>12.10f}'.
                               format(cycle, cycle_energy[-1], delta_energy, diis.norm))
                del cycle_energy[0]

            #diis build extrapolated amplitudes
            if self.diis:
                self.ts, self.td, self.tt = diis.build([self.ts, self.td, self.tt])

    def ccsd(self, code='lambda', silent=True):
        #coupled-cluster single and doubles - lambda and reduced-density matrices

        def update_amplitudes():
            #compute the next cycle amplitudes

            o, v = self.o, self.v

            #intermediates local storage
            Ioo, Ivv = self.intermediates('oo', tilde=False), self.intermediates('vv', tilde=False)
            Iov = self.intermediates('ov', tilde=False)

            Ioooo, Ivvvv = self.intermediates('oooo', tilde=False), self.intermediates('vvvv', tilde=False)
            Iovoo, Iooov = self.intermediates('ovoo', tilde=False), self.intermediates('ooov', tilde=False)
            Iovvo = self.intermediates('ovvo', tilde=False)
            Ivvvo, Ivovv = self.intermediates('vvvo', tilde=False), self.intermediates('vovv', tilde=False)

            Goo, Gvv = self.intermediates('OO', tilde=False), self.intermediates('VV', tilde=False)

            #singles lambda
            l1 = Iov.copy().transpose(1,0)
            l1 += np.einsum('ei,ae->ai', self.ls, Ivv, optimize=True)
            l1 -= np.einsum('am,im->ai', self.ls, Ioo, optimize=True)
            l1 += np.einsum('em,ieam->ai', self.ls, Iovvo, optimize=True)
            l1 += 0.5 * np.einsum('efim,efam->ai', self.ld, Ivvvo, optimize=True)
            l1 -= 0.5 * np.einsum('aemn,iemn->ai', self.ld, Iovoo, optimize=True)
            l1 -= np.einsum('ef,eifa->ai', Gvv, Ivovv, optimize=True)
            l1 -= np.einsum('mn,mina->ai', Goo, Iooov, optimize=True)

            #doubles lambda
            l2 = self.gs[v, v, o, o].copy()

            t = np.einsum('aeij,eb->abij', self.ld, Ivv, optimize=True)
            l2 += t - t.transpose(1,0,2,3)

            t = -np.einsum('abim,jm->abij', self.ld, Ioo, optimize=True)
            l2 += t - t.transpose(0,1,3,2)

            l2 += 0.5 * np.einsum('abmn,ijmn->abij', self.ld, Ioooo, optimize=True)
            l2 += 0.5 * np.einsum('efab,efij->abij', Ivvvv, self.ld, optimize=True)

            t = np.einsum('ei,ejab->abij', self.ls, Ivovv, optimize=True)
            l2 += t - t.transpose(0,1,3,2)

            t = -np.einsum('am,ijmb->abij', self.ls, Iooov, optimize=True)
            l2 += t - t.transpose(1,0,2,3)

            t = np.einsum('aeim,jebm->abij', self.ld, Iovvo, optimize=True)
            l2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

            t = np.einsum('ai,jb->abij', self.ls, Iov, optimize=True)
            l2 += t - t.transpose(1,0,2,3) - t.transpose(0,1,3,2) + t.transpose(1,0,3,2)

            t = np.einsum('ijae,be->abij', self.gs[o, o, v, v], Gvv, optimize=True)
            l2 += t - t.transpose(1,0,2,3)

            t = -np.einsum('imab,mj->abij', self.gs[o, o, v, v], Goo, optimize=True)
            l2 += t - t.transpose(0,1,3,2)

            self.ls = l1*(np.reciprocal(self.deltas[0]).transpose(1,0)) ; self.ld = l2*np.reciprocal(self.deltas[1]).transpose(2,3,0,1)

        def cluster_energy():

            o, v = self.o, self.v

            energy = np.einsum('ia,ai->', self.fs[o, v], self.ls, optimize=True)
            energy += 0.25 * np.einsum('abij,abij->', self.ld, self.gs[v, v, o, o], optimize=True)

            return energy

        def lagrangian_energy():

            o, v = self.o, self.v

            #diagonals of Fock matrix
            Doo = np.diag(np.diag(self.fs[o, o]))
            Dvv = np.diag(np.diag(self.fs[v, v]))

            #singles amplitudes recalculated
            ts  = self.ts * self.deltas[0]
            ts -= np.einsum('ma,mi->ia', self.ts, Doo, optimize=True)
            ts += np.einsum('ie,ae->ia', self.ts, Dvv, optimize=True)

            #doubles amplitudes recalculated
            td = self.td * self.deltas[1]
            t  = np.einsum('imab,mj->ijab', self.td, Doo, optimize=True)
            td += -(t - t.transpose(1,0,2,3))
            t  = np.einsum('ijae,be->ijab', self.td, Dvv, optimize=True)
            td += t - t.transpose(0,1,3,2)

            lagrange = self.cache[0]
            lagrange += np.einsum('ai,ia->', self.ls, ts, optimize=True)
            lagrange += np.einsum('abij,ijab->', self.ld, td, optimize=True)

            return lagrange

        def lambda_perturbative_triples():
            #perturbative triples correction to lambda CCSD

            o, v = self.o, self.v

            #permutations are: i/jk a/bc, i/jk c/ab    and   k/ij a/bc
            permutation_set = [{'ijkabc':+1,'ijkbac':-1,'ijkcba':-1,'jikabc':-1,'jikbac':+1,'jikcba':+1,'kjiabc':-1,'kjibac':+1,'kjicba':+1},
                               {'ijkcab':+1,'ijkacb':-1,'ijkbac':-1,'jikcab':-1,'jikacb':+1,'jikbac':+1,'kjicab':-1,'kjiacb':+1,'kjibac':+1},
                               {'kijabc':+1,'ikjabc':-1,'jikabc':-1,'kijbac':-1,'ikjbac':+1,'jikbac':+1,'kijcba':-1,'ikjcba':+1,'jikcba':+1}]

            lt = np.zeros_like(self.deltas[2])

            #lambda triples
            t = np.einsum('dkbc,adij->ijkabc', self.gs[v, o, v, v], self.ld, optimize=True)
            for i in permutation_set[2]:
                lt += permutation_set[2][i] * np.einsum('ijkabc->' + i, t, optimize=True)
            t = np.einsum('jklc,abil->ijkabc', self.gs[o, o, o, v], self.ld, optimize=True)
            for i in permutation_set[1]:
                lt -= permutation_set[1][i] * np.einsum('ijkabc->' + i, t, optimize=True)
            t = np.einsum('ai,bcjk->ijkabc', self.ls, self.gs[v, v, o, o], optimize=True)
            for i in permutation_set[0]:
                lt += permutation_set[0][i] * np.einsum('ijkabc->' + i, t, optimize=True)
            t = np.einsum('ia,bcjk->ijkabc', self.fs[o,v], self.ld, optimize=True)
            for i in permutation_set[0]:
                lt += permutation_set[0][i] * np.einsum('ijkabc->' + i, t, optimize=True)

            tt = np.zeros_like(lt)

            #t triples
            t = np.einsum('bcdk,ijad->ijkabc', self.gs[v, v, v, o], self.td, optimize=True)
            for i in permutation_set[2]:
                tt += permutation_set[2][i] * np.einsum('ijkabc->' + i, t, optimize=True)
            t = np.einsum('lcjk,ilab->ijkabc', self.gs[o, v, o, o], self.td, optimize=True)
            for i in permutation_set[1]:
                tt -= permutation_set[1][i] * np.einsum('ijkabc->' + i, t, optimize=True)

            tt *= np.reciprocal(self.deltas[2])
            lambda_correction = np.einsum('ijkabc,ijkabc->', lt, tt, optimize=True)/36.0

            return lambda_correction

        def iterator(silent):
            #iterate lambda amplitudes to convergence

            #initialise diis buffers
            if self.diis:
                diis = dii_subspace(self.scf.diis_size, type='c')
                diis.cache = [[self.ls, self.ld]]

            cycle_energy = [cluster_energy()]

            for cycle in range(self.cycles):

                #store pre-update amplitudes
                if self.diis:
                    diis.append([self.ls, self.ld])

                update_amplitudes()
                cycle_energy.append(cluster_energy())

                #test convergence
                delta_energy = np.abs(cycle_energy[-2] - cycle_energy[-1])
                if delta_energy < self.tol:
                    self.converged = True
                    if code != 'lambda(t)':
                        self.cache = [cycle_energy[-1], lagrangian_energy()]
                    else:
                        self.cache = [cycle_energy[-1], lagrangian_energy(), lambda_perturbative_triples()]
                    return
                else:
                    if not silent: print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f}   |diis| = {:>12.10f}'.
                                   format(cycle, cycle_energy[-1], delta_energy, diis.norm))
                    del cycle_energy[0]

                #diis build extrapolated amplitudes
                if self.diis:
                    self.ls, self.ld = diis.build([self.ls, self.ld])

        def oprdm():
            #one-particle reduced density matrix

            o, v, n= self.o, self.v, sum(self.ls.shape)

            gamma = np.zeros((n, n))
            gamma[v, o]  = np.einsum('ai->ai', self.ls, optimize=True)

            gamma[o, v]  =  np.einsum('ia->ia', self.ts, optimize=True)
            gamma[o, v] += np.einsum('bj,ijab->ia', self.ls, self.td, optimize=True)
            gamma[o, v] -= np.einsum('bj,ja,ib->ia', self.ls, self.ts, self.ts, optimize=True)
            gamma[o, v] -= 0.5 * np.einsum('bcjk,ikbc,ja->ia', self.ld, self.td, self.ts, optimize=True)
            gamma[o, v] -= 0.5 * np.einsum('bcjk,jkac,ib->ia', self.ld, self.td, self.ts, optimize=True)

            gamma[v, v]  =  np.einsum('ai,ib->ab', self.ls, self.ts, optimize=True)
            gamma[v, v] += 0.5 * np.einsum('acij,ijbc->ab', self.ld, self.td, optimize=True)

            gamma[o, o]  = -np.einsum('aj,ia->ij', self.ls, self.ts, optimize=True)
            gamma[o, o] -= 0.5 * np.einsum('abjk,ikab->ij', self.ld, self.td, optimize=True)
            gamma[o, o] += np.einsum('ij->ij', np.eye(self.ls.shape[1]), optimize=True)

            self.cache = gamma

        if not self.code in ['ccsd', 'ccsd(t)']:
            self.method(code='ccsd', silent=True)

        if code in ['lambda', 'lambda(t)']:
            self.ls, self.ld = np.zeros_like(self.ts.transpose(1,0)), self.td.transpose(2,3,0,1)
            iterator(silent)
            return self.cache

        if code in 'oprdm':
            try:
                self.ls
            except AttributeError:
                self.ls, self.ld = np.zeros_like(self.ts.transpose(1,0)), self.td.transpose(2,3,0,1)
                iterator(True)

            return oprdm()

class rCC(object):
    #class for Coupled-Cluster methods

    def __init__(self, scf, cycles=50, tol=1e-10, diis=True):
        #initialise CC object

        self.scf = scf
        self.cycles, self.tol, self.diis = cycles, tol, diis

        self.ndocc = self.scf.mol.nele[0]
        self.o, self.v = slice(None, self.ndocc), slice(self.ndocc, None)

        self.code = None

    def method(self, code='ccd', ts=None, td=None, silent=True):
        #execute the method implied by 'code'

        if not code in ['ccd', 'ccsd']:
            print('coupled-cluster code [', code, '] not recognised')
            self.cache = None
            return

        #compute molecular spin Fock and 2-electron repulsion in Hirata Biorthogonal form
        self.f = mos.orbital_transform(self.scf, 'm', self.scf.get('f'))
        self.g = mos.orbital_transform(self.scf, 'm', self.scf.get('i'))
        self.w = 2.0* self.g - self.g.transpose(0,3,2,1)

        o, v = self.o, self.v

        #initial amplitudes
        self.deltas = mos.orbital_deltas(self.scf, 3, mo='x')
        self.ts = np.zeros_like(self.deltas[0])
        self.td = self.g[o, v, o, v].transpose(0,2,1,3) * np.reciprocal(self.deltas[1])

        self.code = code

        func_method = self.update_amplitudes

        self.iterator(func_method, silent)

        return

    def intermediates(self, code):
        #spatial intermediates for Hirata bi-orthogonal 2-electron repulsion integrals

        o, v = self.o, self.v

        if code == 'oo':
            im = np.diag(np.diag(self.f[o, o]))
            im += np.einsum('iemf,jmef->ij',  self.w[o, v, o, v], self.td, optimize=True)
            im += np.einsum('iemf,je,mf->ij', self.w[o, v, o, v], self.ts, self.ts, optimize=True)

        if code == 'vv':
            im = np.diag(np.diag(self.f[v, v]))
            im -= np.einsum('mbne,mnae->ab',  self.w[o, v, o, v], self.td, optimize=True)
            im -= np.einsum('mbne,ma,ne->ab', self.w[o, v, o, v], self.ts, self.ts, optimize=True)

        if code == 'ov':
            im = np.einsum('iame,me->ia', self.w[o, v, o, v], self.ts, optimize=True)

        if code == 'OO':
            im = self.intermediates('oo')
            im += np.einsum('meij,me->ij', self.w[o, v, o, o], self.ts, optimize=True)

        if code == 'VV':
            im = self.intermediates('vv')
            im += np.einsum('meab,me->ab', self.w[o, v, v, v], self.ts, optimize=True)

        if code == 'oooo':
            im = self.g[o, o, o, o].copy().transpose(0,2,1,3)
            im += np.einsum('jeik,le->ijkl', self.g[o, v, o, o], self.ts, optimize=True)
            im += np.einsum('iejl,ke->ijkl', self.g[o, v, o, o,], self.ts, optimize=True)
            im += np.einsum('iejf,klef->ijkl', self.g[o, v, o, v], self.td, optimize=True)
            im += np.einsum('iejf,ke,lf->ijkl', self.g[o, v, o, v], self.ts, self.ts, optimize=True)

        if code == 'vvvv':
            im = self.g[v, v, v, v].copy().transpose(0, 2, 1, 3)
            im -= np.einsum('mdac,mb->abcd', self.g[o, v, v, v], self.ts, optimize=True)
            im -= np.einsum('mcbd,ma->abcd', self.g[o, v, v, v], self.ts, optimize=True)

        if code == 'voov':
            im = self.g[o, v, v, o].copy().transpose(2, 0, 3, 1)
            im -= np.einsum('ibmj,ma->aijb', self.g[o, v, o, o], self.ts, optimize=True)
            im += np.einsum('ibae,je->aijb', self.g[o, v, v, v], self.ts, optimize=True)
            im -= 0.5 * np.einsum('meib,jmea->aijb', self.g[o, v, o, v], self.td, optimize=True)
            im -= np.einsum('meib,je,ma->aijb', self.g[o, v, o, v], self.ts, self.ts, optimize=True)
            im += 0.5 * np.einsum('meib,jmae->aijb', self.w[o, v, o, v], self.td, optimize=True)

        if code == 'vovo':
            im = self.g[o, o, v, v].copy().transpose(2, 0, 3, 1)
            im -= np.einsum('mbij,ma->aibj', self.g[o, v, o, o], self.ts, optimize=True)
            im += np.einsum('ieab,je->aibj', self.g[o, v, v, v], self.ts, optimize=True)
            im -= 0.5 * np.einsum('mbie,jmea->aibj', self.g[o, v, o, v], self.td, optimize=True)
            im -= np.einsum('mbie,je,ma->aibj', self.g[o, v, o, v], self.ts, self.ts, optimize=True)

        return im

    def update_amplitudes(self):
        #next cycle amplitude computation

        o, v = self.o, self.v

        #singles amplitudes

        if self.code == 'ccsd':
            Ioo, Ivv, Iov = self.intermediates('oo'), self.intermediates('vv'), self.intermediates('ov')

            t1  = np.einsum('ae,ie->ia', Ivv - np.diag(np.diag(self.f[v, v])), self.ts, optimize=True)
            t1 -= np.einsum('mi, ma->ia', Ioo - np.diag(np.diag(self.f[o, o])), self.ts, optimize=True)
            t1 += 2.0 * np.einsum('kc,kica->ia', Iov, self.td, optimize=True)
            t1 -= np.einsum('me,imea->ia', Iov, self.td, optimize=True)
            t1 += np.einsum('me,ie,ma->ia', Iov, self.ts, self.ts, optimize=True)
            t1 += np.einsum('meai,me->ia', self.w[o, v, v, o], self.ts, optimize=True)
            t1 += np.einsum('mfae,imef->ia', self.w[o, v, v, v], self.td, optimize=True)
            t1 += np.einsum('mfae,mf,ie->ia', self.w[o, v, v, v], self.ts, self.ts, optimize=True)
            t1 -= np.einsum('nemi,mnae->ia', self.w[o, v, o, o], self.td, optimize=True)
            t1 -= np.einsum('nemi,ne,ma->ia', self.w[o, v, o, o], self.ts, self.ts, optimize=True)
        else:
            t1 = self.ts

        #doubles amplitudes
        IOO, IVV = self.intermediates('OO'), self.intermediates('VV')

        Ioooo, Ivoov = self.intermediates('oooo'), self.intermediates('voov')
        Ivovo, Ivvvv = self.intermediates('vovo'), self.intermediates('vvvv')

        t2 =  0.5 * self.g[o, v, o, v].transpose(0,2,1,3).copy()
        t2 += 0.5 * np.einsum('mnij,mnab->ijab', Ioooo, self.td, optimize=True)
        t2 += 0.5 * np.einsum('mnij,ma,nb->ijab', Ioooo, self.ts, self.ts, optimize=True)
        t2 += 0.5 * np.einsum('abef,ijef->ijab', Ivvvv, self.td, optimize=True)
        t2 += 0.5 * np.einsum('abef,ie,jf->ijab', Ivvvv, self.ts, self.ts, optimize=True)

        t2 += np.einsum('ae,ijeb->ijab', IVV - np.diag(np.diag(self.f[v, v])), self.td, optimize=True)
        t2 -= np.einsum('mi,mjab->ijab', IOO - np.diag(np.diag(self.f[o, o])), self.td, optimize=True)

        t2 += np.einsum('iaeb,je->ijab', self.g[o, v, v, v], self.ts, optimize=True)
        t2 -= np.einsum('mibe,ma,je->ijab', self.g[o, o, v, v], self.ts, self.ts, optimize=True)
        t2 -= np.einsum('iajm,mb->ijab', self.g[o, v, o, o], self.ts, optimize=True)
        t2 -= np.einsum('iaem,je,mb->ijab', self.g[o, v, v, o], self.ts, self.ts, optimize=True)

        t2 += 2 * np.einsum('amie,mjeb->ijab', Ivoov, self.td, optimize=True)
        t2 -= np.einsum('amei,mjeb->ijab', Ivovo, self.td, optimize=True)
        t2 -= np.einsum('amie,mjbe->ijab', Ivoov, self.td, optimize=True)
        t2 -= np.einsum('bmei,mjae->ijab', Ivovo, self.td, optimize=True)

        self.ts = t1 * np.reciprocal(self.deltas[0])
        self.td = (t2 + t2.transpose(1,0,3,2)) * np.reciprocal(self.deltas[1])

    def cluster_energy(self):
        #compute the coupled-cluster energy correction

        o, v = self.o, self.v

        energy  = np.einsum('iajb,ijab->', self.w[o, v, o, v], self.td, optimize=True)
        energy += np.einsum('iajb,ia,jb->', self.w[o, v, o, v], self.ts, self.ts, optimize=True)

        return energy

    def iterator(self, func, silent):
        #consistent field coupled-cluster iterations

        #initialise diis buffers
        if self.diis:
            diis = dii_subspace(self.scf.diis_size, type='c')
            diis.cache = [[self.ts, self.td]]

        cycle_energy = [self.cluster_energy()]
        assert np.isclose(cycle_energy[-1], MP(self.scf, 'MP2').correction)

        for cycle in range(self.cycles):

            #store pre-update amplitudes
            if self.diis:
                diis.append([self.ts, self.td])

            func()

            #calculate current cycle energy
            cycle_energy.append(self.cluster_energy())

            #test convergence
            delta_energy = np.abs(cycle_energy[-2] - cycle_energy[-1])
            if delta_energy < self.tol:
                self.converged = True
                self.cache = [cycle_energy[-1]]
                return
            else:
                if not silent:
                    if self.diis:
                        print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f}   |diis| = {:>12.10f}'.
                        format(cycle, cycle_energy[-1], delta_energy, diis.norm))
                    else:
                        print('cycle = {:>3d}  energy = {:>15.10f}   \u0394E = {:>12.10f} '.
                        format(cycle, cycle_energy[-1], delta_energy))
                del cycle_energy[0]

            #diis build extrapolated amplitudes
            if self.diis:
                self.ts, self.td = diis.build([self.ts, self.td])
