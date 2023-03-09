from __future__ import division

from mol.mol import molecule
from scf.rhf import RHF
from mol.mol import CONSTANTS

if __name__ == '__main__':
    mol = molecule([['O', (0.0, 0.0, 0.0)], 
                    ['H', (0,-0.757 ,0.587)], 
                    ['H', (0, 0.757 ,0.587)]], 
                    spin=0,
                    units='angstrom',
                    charge=0,
                    gto='3-21g',
                    silent=True)

    scf = RHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    #***********************
    #*  Quadrupole Tensor  *
    #***********************

    #get 6 components of symmetric quadrupole tensor
    from int.aello import aello
    import numpy as np

    quadrupole = -np.array(aello(scf.mol.atom, scf.mol.orbital, 'quadrupole', scf.get('d'), scf.mol.gauge)) 

    cartesian_decode = {0 : ['xx',0,0], 1 : ['yy',1,1], 2 : ['zz',2,2], 3 : ['xy',0,1], 4 : ['yz',1,2], 5 : ['zx',2,0]}

    #electronic component
    q_mu = 2.0 * np.einsum('ir,xri->x', scf.get('d'), quadrupole) 

    #nuclear contribution
    for i in range(3):
        for j in scf.mol.atom:
            q_mu[i] += j.center[cartesian_decode[i][1]] * j.center[cartesian_decode[i][2]] * j.number

    #units are Debye Angstrom
    q_mu *= CONSTANTS('au->debye') * CONSTANTS('bohr->angstrom')

    print('\nQuadrupole Tensor\n---------------------------')
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(q_mu[0], q_mu[3], q_mu[5]))
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(q_mu[3], q_mu[1], q_mu[4]))
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(q_mu[5], q_mu[4], q_mu[2]))

    #*******************************
    #* Traceless Quadrupole Tensor *
    #*******************************

    #traceless tensor form   Q_ij^' = Q_ij - Q \delta_ij / 3
    traceless_tensor = np.zeros((3, 3))

    #off-diagonal unchanged and symmetrize
    traceless_tensor[0,1], traceless_tensor[1,2], traceless_tensor[0,2] = q_mu[3:]
    traceless_tensor += traceless_tensor.transpose()

    trace = sum(q_mu)/3.0
    for i in range(3): traceless_tensor[i,i] = q_mu[i] - trace

    print('\nTraceless Quadrupole Tensor\n---------------------------')
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(traceless_tensor[0,0], traceless_tensor[0,1], traceless_tensor[0,2]))
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(traceless_tensor[1,0], traceless_tensor[1,1], traceless_tensor[1,2]))
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(traceless_tensor[2,0], traceless_tensor[2,1], traceless_tensor[2,2]))

    #*********************
    #* Principal Moments *
    #*********************

    #eigenvalues - principal moments - order >
    eig = np.linalg.eigvals(traceless_tensor)
    eig = eig[np.argsort(eig)][::-1]

    print('\nPrincipal Moments\n---------------------------')
    print(' {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(eig[0], eig[1], eig[2]))

    #asymmetry defined as (P_xx - P_yy) / P_ZZ where P_zz > P_xx > P_yy  
    amplitude = np.linalg.norm(eig)
    asymmetry = (eig[1] - eig[2])/eig[0]
    print('\nAmplitude  {:>7.4f}       Asymmetry Factor {:>7.4f}'.format(amplitude, asymmetry))



# nuclear repulsion      9.1882584177
# total electronic     -84.7736598020

# final total energy   -75.5854013843

# Quadrupole Tensor
# ---------------------------
#  -6.8200  -0.0000  -0.0000
#  -0.0000  -4.1338   0.0000
#  -0.0000   0.0000  -5.2361

# Traceless Quadrupole Tensor
# ---------------------------
#  -1.4233  -0.0000  -0.0000
#  -0.0000   1.2628   0.0000
#  -0.0000   0.0000   0.1605

# Principal Moments
# ---------------------------
#   1.2628   0.1605  -1.4233

# Amplitude   1.9095       Asymmetry Factor  1.2542
