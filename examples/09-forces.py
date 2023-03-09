from mol.mol import molecule
from scf.rhf import RHF
import int.mo_spin as mos
from phf.eig import solver
from mol.mol import CONSTANTS

import numpy as np

if __name__ == '__main__':
    #we're using the geometry from the Crawford Projects
    mol = molecule([['O', ( 0.000000000000, -0.143225816552, 0.000000000000)], 
                    ['H', ( 1.638036840407,  1.136548822547 ,0.000000000000)], 
                    ['H', (-1.638036840407,  1.136548822547 ,0.000000000000)]], 
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='sto-3g',
                    silent=False)

    scf = RHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    scf.analyse('geometry')
    '''
    aello has capability to calculate gradient integrals, it returns the resultant RHF force as a [natm, 3] tensor.
    So [1,2] is the force in the z-axis direction for atom 2. Remember forces are negative of gradient. UNits are $\frac{Eh}{bohr}$
    '''
    from int.aello import aello_dx

    force_tensor = aello_dx(scf.mol.atom, scf.mol.orbital, scf.get('d'), scf.get('f'))

    print('\nRHF Forces')
    print('   atom        x               y              z\n---------------------------------------------------')
    for i, atm in enumerate(scf.mol.atom):
        print('{:2}  {:2}   {:>12.8f}   {:>12.8f}   {:>12.8f} '.format(atm.id, atm.symbol, 
                                                                force_tensor[i, 0], force_tensor[i,1], force_tensor[i,2]))

    '''
    we can get the force of the electronic component by subtracting the nuclear component
    '''
    nuclear_gradient = np.zeros((scf.mol.natm, 3))

    for axis in range(3):
        for i, p in enumerate(scf.mol.atom):
            for q in scf.mol.atom:
                r = np.linalg.norm(p.center - q.center)

                if r != 0.0: nuclear_gradient[i, axis] += (p.center[axis] - q.center[axis]) * p.number * q.number/pow(r, 3)

    electronic_tensor = force_tensor - nuclear_gradient
    print('\nelectronic Forces')
    print('   atom        x               y              z\n---------------------------------------------------')
    for i, atm in enumerate(scf.mol.atom):
        print('{:2}  {:2}   {:>12.8f}   {:>12.8f}   {:>12.8f} '.format(atm.id, atm.symbol, 
                                                                electronic_tensor[i, 0], electronic_tensor[i,1], electronic_tensor[i,2]))

    '''
    the force on the Oxygen is directed towards the midpoint of the hydrogens and the hydrogens are attracted towards the oxygen roughly
    along the bonds. We can compute the gradient by numerical means as $\frac{dE}{dx} = \frac{E[x] - E[x+dx]}{dx}...
    '''
    #change geometry + dy
    dy = 1e-4
    scf.mol.atom[0].center[1] += dy
    scf.mol.silent = True

    energy_at_plus_dy = scf.execute()

    #change geometry -dy
    scf.mol.atom[0].center[1] -= 2 * dy
    energy_at_minus_dy = scf.execute()

    #using central-difference evaluation of gradient
    force = -(energy_at_plus_dy - energy_at_minus_dy)/ (2 * dy)
    print('\nNumerical differentiation of RHF energy for O atom in y-direction is {:>12.8f} '.format(force))

    '''
    one major use of forces is in Born-Oppenheimer Molecular Dynamics BOMD. This uses the forces (accelerations) to determine where the
    geometry wants to move to for a number of time steps.
    '''
    mol = molecule([['H', ( 0.000000000000, 0.000000000000, 0.000000000000)], 
                    ['H', ( 0.740000000000, 0.000000000000 ,0.000000000000)]], 
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='sto-3g',
                    silent=True)

    scf = RHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=False)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    from mol.utl import get_atomic_weight

    #number of time steps and step duration - time in femtoseconds
    steps, dt, method = 200, 5, 'velocity-verlet'
    cache_size = 2 if method == 'velocity-verlet' else 3

    #arrays fro computed properties
    time_energy = np.zeros((steps, 2))
    geometry    = np.zeros((steps, scf.mol.natm, 3))
    velocity = np.zeros((scf.mol.natm, 3))

    #initial values - initial velocity is zero
    time_energy[0, 1] = scf_energy
    geometry[0, :, :] = [p.center[:] for p in scf.mol.atom]

    #forward propogation 
    force_cache = []
    force_cache.append(aello_dx(scf.mol.atom, scf.mol.orbital, scf.get('d'), scf.get('f')))

    #intergrators are provided in utl module as
    from mol.utl import integrator
    #methods available are velocity-verlet, Beeman and Adams-Moulton - default is velocity-verlet

    for cycle in range(1, steps):

        #update geometry - elementry school $s = ut + \frac{1}{2} f t^2$ - convert reative atomic weight to amu
        for i, p in enumerate(scf.mol.atom):
            geometry[cycle, i, :] = geometry[cycle-1, i, :] + velocity[i, :] * dt + 0.5 * dt * dt * force_cache[-1][i, :] / get_atomic_weight(p.number) 

        #get energy and forces at new position
        for i, p in enumerate(scf.mol.atom):
            p.center[:] = geometry[cycle, i, :]

        scf_energy = scf.execute()

        time_energy[cycle, 0] = cycle * dt
        time_energy[cycle, 1] = scf_energy

        #update cache
        force_cache.append(aello_dx(scf.mol.atom, scf.mol.orbital, scf.get('d'), scf.get('f')))

        if len(force_cache) > cache_size: del force_cache[0]

        #apply integrator for new velocity - 1st cycle always velocity-verlet
        cycle_method = 'velocity-verlet' if cycle == 1 else method
        for i in range(scf.mol.natm):
            velocity[i, :] += integrator(scf, i, dt, force_cache, cycle_method)

    '''
    we now have the position of the atoms at each step in the time propogation as determined by the forces exerted on the atoms. Having
    the geometry we can compute any bond length or angle in that geometry, thus we can determine how these features change during the 
    time period. We can, for example, plot the vibration of the HF bond and from the plot read the frequency of vibration.
    '''
    vibration = np.zeros((steps))
    for t in range(steps):
        vibration[t] = np.linalg.norm(geometry[t, 0, :] - geometry[t, 1, :])

    import matplotlib.pyplot as plt   

    time_energy[:,0] *= CONSTANTS('au->femtosecond')
    vibration *=  CONSTANTS('bohr->angstrom')

    plt.figure()
    plt.subplot(211)
    plt.xlabel('t (fs)')
    plt.ylabel('E (au)')
    plt.grid(True)
    plt.title('$H_2$ bond length vibrational energy')
    plt.plot(time_energy[:,0], time_energy[:,1])

    plt.subplot(212)
    plt.xlabel('t (fs)')
    plt.ylabel('r ($\AA$)')
    plt.grid(True)
    plt.plot(time_energy[:,0], vibration)

    plt.tight_layout()
    plt.show()

    '''
    we can see that the bond length at minimum energy is about 0.71 anstroms, while the extremes of the bond length range occur at energy maximums.
    we can get the optimized bond length as...
    '''
    min_energy_idx = np.argmin(time_energy[:, 1])
    print('\nminimum energy is {:>12.8f} au at time {:<5.3f} fs'.format(time_energy[:, 1][min_energy_idx], time_energy[:, 0][min_energy_idx]))
    print('bond length at minimum energy is {:>7.5f} \u212B'.format(vibration[min_energy_idx]))

    '''
    this gives an optimized bond length of 0.71224 A which is agreement with a Gaussian geometry optimization to be found here
    [https://www.tau.ac.il/~ephraim/gauss-out-opt-H2.pdf]. We can find the vibrational frequency of he HF bond by calculating the frequency, the period is about
    6 fs so vibrational frequency is roughly 1/6c or about 5000 cm^-1 - the experimental value is 4342.
    '''

'''
******************
*   scf output   *
******************
method                  RHF 
charge                  0 
spin                    0 
units                   bohr
open shell              False      :  multiplicity  1

basis is                sto-3g
analytic integration    aello cython - McMurchie-Davidson scheme

diis                    True  : buffer size is  6
scf control             maximum cycles are  50    :  convergence tolerance  1e-10

basis analysis
--------------
shell   l    pGTO    cGTO
-------------------------
O    0 
  0     0      3       1
  1     0      3       1
  2     1      3       1
H    1 
  3     0      3       1
H    2 
  4     0      3       1

-------------------------
number of shells            5
number of primative GTO    21
number of contracted GTO    7

 cycle        E SCF              ΔE            |diis|               homo            lumo
------------------------------------------------------------------------------------------------
    1     -78.28658323       7.8287e+01      8.4510e-01          0.30910705      0.55914745   
    2     -84.04831633       5.7617e+00      3.5580e-01         -0.53724397      0.40720333   
    3     -82.71696597       1.3314e+00      9.2429e-02         -0.34937966      0.49506621   
    4     -82.98714079       2.7017e-01      1.9368e-02         -0.39652094      0.47684760   
    5     -82.93813321       4.9008e-02      8.1314e-03         -0.38696302      0.47890075   
    6     -82.94398045       5.8472e-03      3.4757e-03         -0.38752441      0.47760521   
    7     -82.94444058       4.6013e-04      2.5198e-05         -0.38758528      0.47762524   
    8     -82.94444697       6.3936e-06      4.7658e-06         -0.38758674      0.47761876   
    9     -82.94444702       4.1294e-08      3.2206e-08         -0.38758674      0.47761872   
   10     -82.94444702       1.4070e-10      9.4176e-11         -0.38758674      0.47761872   
   11     -82.94444702       0.0000e+00      1.8538e-14         -0.38758674      0.47761872   
   12     -82.94444702       2.8422e-14      2.0764e-14         -0.38758674      0.47761872   

nuclear repulsion      8.0023670618
total electronic     -82.9444470159

final total energy   -74.9420799540

Geometry
-----------
 O       0.0000    -0.1432     0.0000 
 H       1.6380     1.1365     0.0000 
 H      -1.6380     1.1365     0.0000 

RHF Forces
   atom        x               y              z
---------------------------------------------------
 0  O      0.00000000     0.09744138     0.00000000 
 1  H     -0.08630006    -0.04872069     0.00000000 
 2  H      0.08630006    -0.04872069     0.00000000 

electronic Forces
   atom        x               y              z
---------------------------------------------------
 0  O      0.00000000     2.37714778     0.00000000 
 1  H     -1.63841918    -1.18857389     0.00000000 
 2  H      1.63841918    -1.18857389     0.00000000 

Numerical differentiation of RHF energy for O atom in y-direction is   0.09744138 

minimum energy is  -1.11750589 au at time 16.690 fs
bond length at minimum energy is 0.71224 Å
'''