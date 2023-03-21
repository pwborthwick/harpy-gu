import numpy as np

from mol.mol import molecule, CONSTANTS
from scf.rhf import RHF

if __name__ == '__main__':

    '''
    Here we will look at the spectrum of the transition dipoles calculated by the RT-TDHF
    module using the Magnus second order propagator. The geometry we're using is from this
    NWCHEM article [https://nwchemgit.github.io/RT-TDDFT.html#Hints_and_Tricks].
    '''

    mol = molecule([['O', (  0.00000000,    -0.00001441,    -0.34824012)],
                    ['H', (  0.00000000,     0.76001092,    -0.93285191)],
                    ['H', (  0.00000000,    -0.75999650,    -0.93290797)]],
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='6-31g',
                    silent=False)

    scf = RHF(mol,
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    '''
    First we define a Gaussian envelope for use with the spectrum analysis and an instantaneous 'kick'
    to apply at time zero to start the propagation.
    '''

    def gaussian(t):
        #Gaussian waveshape

        rho =0.1
        field_strength = 0.0001

        return np.exp(-(t**2)/ (2 * rho * rho)) * field_strength

    def kick(t):
        #instantaneuos pulse

        field_strength = 0.0001
        return 0.0 if t != 0 else field_strength

    '''
    We will use the RT_TDHF class from the phf.tdhf module and to analyse the dipole propagation we will use
    spectrum and peaks from the mol.utl module. The propagation is followed for 1000 steps with a 0.2 atomic
    time unit step size. This gives a total elapsed time of 200 atomic time units (~4.8 fs). The dipole
    propagation is then analysed using Pade approximants in the get_spectrum routine. Each polarization
    direction is considered in turn and the combined normalised plot displayed.
    '''

    from phf.tdhf import TDHF, RT_TDHF
    from mol.utl import get_spectrum, peaks
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5,2.5))
    plt.title('Magnus 2   $H_2O$ absorption components')
    plt.grid()

    frequency, spectrum, ev_peaks = [], [], []
    plot_color, polarization = ['r', 'b', 'orange'], ['x', 'y', 'z']

    for axis in polarization:
        rt = RT_TDHF(scf, pulse=kick, dt=0.2, cycles=1000, axis=axis)
        rt.execute('magnus')

        f, s = get_spectrum(rt.cache[:,0], rt.cache[:,2], damping=50.0, points= 5000, interval=1.0,
                            tick=0.0001, field=[gaussian, 'i', 0.0001])
        f *= CONSTANTS('hartree->eV')
        frequency.append(f)
        spectrum.append(s)
        ev_peaks.append(peaks(s, f, 0.01))

    scale = max(np.max(spectrum[0]), np.max(spectrum[1]), np.max(spectrum[2]))
    for axis in range(3):
        plt.plot(frequency[axis], spectrum[axis]/scale,
                 label=polarization[axis], color=plot_color[axis])

    plt.legend(loc=1)
    plt.xlabel('Energy (eV)')
    plt.ylabel('scaled $\sigma(\omega)$ [arb. units]')

    plt.show()

    '''
    This plot should look the same as the one labeled 'Water Gas-Phase 6-31G/TD-PBE0 Polarization Dependent Absorption'
    from the reference above. We can also plot the combined absorption...
    '''

    plt.figure(figsize=(5,2.5))
    plt.title('Magnus 2   $H_2O$ absorption ')
    plt.xlabel('Energy (eV)')
    plt.ylabel('scaled $\sigma(\omega)$ [arb. units]')
    plt.grid()
    resultant = sum(spectrum)
    resultant /= np.max(resultant)
    plt.plot(frequency[0], resultant, color='k')

    plt.show()

    '''
    We have used peaks from mol.utl in order to find the energies at which maximum absorption occurs and we can compare with the
    analytical results from the TDHF class. We have rejected roots with a zero'ish oscillator strength and restricted the energy
    range to our RT range. The difference between the two methods is less than 1%.
    '''

    td = TDHF(scf)

    peak_count = [0, 0, 0]
    print('\nAbsorption Peaks from TDHF and RT-TDHF\n--------------------------------------\n')
    print(' root    polarization    TDA-TDHF(eV)  RT-TDHF(eV)')
    print('--------------------------------------------------')
    for root, values in enumerate(td.cache):
        if values['energy'][1] > 20: break
        if values['electric length'][1] < 1e-10: continue

        axis = np.argmax(np.abs(values['electric length'][0]))
        print(' {:2}            {:1}            {:>6.2f}       {:>6.2f}'.
                                                format(sum(peak_count)+1, polarization[axis],
                                                values['energy'][1], ev_peaks[axis][peak_count[axis]]))

        peak_count[axis] += 1

'''
******************
*   scf output   *
******************
method                  RHF
charge                  0
spin                    0
units                   angstrom
open shell              False      :  multiplicity  1

basis is                6-31g
analytic integration    aello cython - McMurchie-Davidson scheme

diis                    True  : buffer size is  6
scf control             maximum cycles are  50    :  convergence tolerance  1e-10

basis analysis
--------------
shell   l    pGTO    cGTO
-------------------------
O    0
  0     0      6       1
  1     0      1       1
  2     0      3       1
  3     1      3       1
  4     1      1       1
H    1
  5     0      1       1
  6     0      3       1
H    2
  7     0      1       1
  8     0      3       1

-------------------------
number of shells            9
number of primative GTO    30
number of contracted GTO   13

 cycle        E SCF              ΔE            |diis|               homo            lumo
------------------------------------------------------------------------------------------------
    1     -68.37035413       6.8370e+01      2.5764e+00          0.57486617      0.58126340
    2     -98.27799718       2.9908e+01      2.3735e+00         -1.69194400     -0.32895257
    3     -75.71338437       2.2565e+01      2.1823e+00          0.25595646      0.32275623
    4     -91.53692279       1.5824e+01      1.6447e+00         -1.08555032     -0.02368248
    5     -81.14072166       1.0396e+01      1.1587e+00         -0.13946337      0.26262306
    6     -85.25250064       4.1118e+00      7.0528e-01         -0.49671486      0.19809725
    7     -85.15580309       9.6698e-02      2.7522e-02         -0.49846996      0.20326623
    8     -85.16131887       5.5158e-03      4.2573e-03         -0.50083395      0.20342390
    9     -85.16232606       1.0072e-03      5.4486e-04         -0.50113630      0.20349958
   10     -85.16238873       6.2672e-05      7.2136e-05         -0.50112820      0.20348634
   11     -85.16236177       2.6967e-05      1.0999e-05         -0.50112057      0.20348549
   12     -85.16235964       2.1301e-06      2.0535e-06         -0.50111983      0.20348550
   13     -85.16235899       6.4581e-07      4.3672e-07         -0.50111960      0.20348547
   14     -85.16235913       1.3809e-07      3.8126e-08         -0.50111961      0.20348546
   15     -85.16235908       4.4805e-08      7.5647e-09         -0.50111960      0.20348546
   16     -85.16235909       7.1300e-09      1.7884e-09         -0.50111960      0.20348546
   17     -85.16235909       1.5217e-10      1.5530e-10         -0.50111960      0.20348546
   18     -85.16235909       4.9269e-11      3.4059e-11         -0.50111960      0.20348546
   19     -85.16235909       5.5991e-12      6.1972e-12         -0.50111960      0.20348546

nuclear repulsion      9.1782622135
total electronic     -85.1623590905

final total energy   -75.9840968770

Absorption Peaks from TDHF and RT-TDHF
--------------------------------------

 root    polarization    TDA-TDHF(eV)  RT-TDHF(eV)
--------------------------------------------------
  1            x              9.41         9.39
  2            z             11.83        11.79
  3            y             13.91        13.97
  4            y             15.54        15.52
  5            z             19.36        19.15
'''
