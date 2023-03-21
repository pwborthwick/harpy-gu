import numpy as np

from mol.mol import molecule, CONSTANTS
from scf.rhf import RHF

from phf.tdhf import TDHF, RT_TDHF

if __name__ == '__main__':

    '''
    In order to look at the rotary stengths we will change to the molecule (S)-methyloxirane in a minaimal basis
    '''

    mol = molecule([['C',  ( 0.152133, -0.035800,  0.485797)],
                    ['C',  (-1.039475,  0.615938, -0.061249)],
                    ['C',  ( 1.507144,  0.097806, -0.148460)],
                    ['O',  (-0.828215, -0.788248, -0.239431)],
                    ['H',  ( 0.153725, -0.249258,  1.552136)],
                    ['H',  (-1.863178,  0.881921,  0.593333)],
                    ['H',  (-0.949807,  1.214210, -0.962771)],
                    ['H',  ( 2.076806, -0.826189, -0.036671)],
                    ['H',  ( 2.074465,  0.901788,  0.325106)],
                    ['H',  ( 1.414895,  0.315852, -1.212218)]],
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='sto-3g',
                    silent=True)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    print('(S)-Methyloxirane energy is {:>14.8f} hartree'.format(scf_energy))

    '''
    We can get the rotatory strengths in both length and velocity gauges from the tdhf cache. Note electric
    properties are calculated in the charge center gauge and magnetic length transition dipole is evaluated
    in the origin gauge.
    '''

    td = TDHF(scf)

    print('\nRotatory Strengths from TDHF\n----------------------------\n')
    print(' root  excitation(eV)  Gauge (L)        (V)  ')
    print('-----------------------------------------------')

    roots = 5
    for root in range(roots):
        excitation_energy = td.cache[root]['energy']
        rotatory_length   = td.cache[root]['rotatory length']
        rotatory_velocity = td.cache[root]['rotatory velocity']

        print('   {:1}     {:>9.6f}       {:>9.6f}  {:>9.6f}  '.format(root+1, excitation_energy[1], rotatory_length, rotatory_velocity))

    '''
    Having got the rotatory strengths we can look at optical properties like the one-photon absorption  (OPA) and
    the electronic circular dichroism (ECD). The TDHF class has a circular_dichroism method with parameters - the
    number of excited states, the method either 'opa'|'ecd', the units either 'eV'|'nm' and the broadening shape
    definition as a string 'wave shape:rho:points'. The wave shape can be either 'gaussian'|'lorentzian', rho for
    Gaussian is the standard deviation of the shape and for Lorentzian rho is the full-width at half-maximum. Points
    are the number of x-axis points (extrapolated) to consider.
    '''

    units, wave, method = 'nm', 'gaussian', 'ecd'
    (x,y), (p,b) = td.circular_dichroism(5, method=method, units=units, shape= wave + ':0.01:5000')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,2.5), layout='constrained')

    #axis labels
    if units == 'nm':
        ax.set_xlabel('\u03BB [nm]')
    if units == 'eV':
        ax.set_xlabel('E [eV]')
    if method == 'opa': ax.set_ylabel('\u03B5 [L $mol^{-1}$ $cm^{-1}$')
    if method == 'ecd': ax.set_ylabel('\u0394\u03B5 [L $mol^{-1}$ $cm^{-1}$')

    plt.grid()
    plt.title('(S)-methyloxirane - ' + wave.capitalize())

    #compute a reasonable width for bars
    width = (np.max(p) - np.min(p))*0.01

    #sticks and broadening
    ax.bar(p, b, width=width, color='k')
    ax.plot(x, y, color='orange')

    plt.show()

'''
(S)-Methyloxirane energy is  -189.51341726 hartree

Rotatory Strengths from TDHF
----------------------------

 root  excitation(eV)  Gauge (L)        (V)
-----------------------------------------------
   1     10.385901        0.004605   0.001748
   2     11.893340       -0.001470  -0.002128
   3     13.790461       -0.020385  -0.022571
   4     14.325934       -0.007912   0.003880
   5     15.356235       -0.042587  -0.044122
'''
