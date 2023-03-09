from mol.mol import molecule
from scf.uhf import UHF
from scf.rhf import RHF

if __name__ == '__main__':
    mol = molecule([['O', (0.0, 0.0, 0.0)], 
                    ['H', (0,-0.757 ,0.587)], 
                    ['H', (0, 0.757 ,0.587)]], 
                    spin=1,
                    units='angstrom',
                    charge=1,
                    gto='3-21g',
                    silent=False)

    scf = UHF(mol, 
              cycles=50,
              tol=1e-8,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')
    
    scf.analyse('geometry,dipole,charge,bonds')
  
    #********************
    #* check with PySCF *
    #********************
    import numpy as np
    print('\nPySCF check is ',np.isclose(scf_energy, -75.1983066869163))

    #********************
    #* Closed Shell UHF *
    #********************
    #Normally a spin=0 UHF will converge to a lower energy than
    #the corresponding RHF computation. However, a RHF solution 
    #can be forced by using the .closed_shell = 'r' property.

    #change mol values for spin=0
    scf.mol.spin, scf.mol.charge, scf.mol.silent = 0, 0, True
    #enforce RHF solution
    scf.closed_shell = 'r'
    #UHF solver
    uhf_energy = scf.execute()

    rhf = RHF(mol, 
              cycles=50,
              tol=1e-8,
              diis=True)

    #RHF solver
    rhf_energy = rhf.execute()

    #assert UHF(spin=0, RHF solution) = RHF
    assert uhf_energy, rhf_energy

'''
*******************************************************************************************************************************
                                                           OUTPUT                                                            *
*******************************************************************************************************************************
******************
*   scf output   *
******************
method                  UHF 
charge                  1 
spin                    1 
units                   angstrom
open shell              True      :  multiplicity  2

basis is                3-21g
analytic integration    aello cython - McMurchie-Davidson scheme

diis                    True  : buffer size is  6
scf control             maximum cycles are  50    :  convergence tolerance  1e-08

 cycle        E SCF              ΔE            |diis|               S²       multiplicity
------------------------------------------------------------------------------------------------------
    1     -71.23057360       7.1231e+01      4.9853e+00           1.325          2.510  
    2     -91.67004266       2.0439e+01      2.9293e+00           0.761          2.011  
    3     -80.84032662       1.0830e+01      2.0953e+00           0.754          2.004  
    4     -85.77378335       4.9335e+00      8.3040e-01           0.752          2.002  
    5     -83.85566428       1.9181e+00      3.3511e-01           0.753          2.003  
    6     -84.37661162       5.2095e-01      1.2555e-01           0.753          2.003  
    7     -84.38760756       1.0996e-02      7.4714e-03           0.754          2.004  
    8     -84.38623015       1.3774e-03      1.6573e-03           0.755          2.005  
    9     -84.38664965       4.1950e-04      3.9609e-04           0.755          2.005  
   10     -84.38655281       9.6847e-05      7.2488e-05           0.755          2.005  
   11     -84.38656358       1.0777e-05      1.2502e-05           0.755          2.005  
   12     -84.38656458       1.0015e-06      3.3863e-06           0.755          2.005  
   13     -84.38656517       5.8953e-07      5.4072e-07           0.755          2.005  
   14     -84.38656507       1.0433e-07      1.0872e-07           0.755          2.005  
   15     -84.38656511       4.1687e-08      4.8450e-08           0.755          2.005  
   16     -84.38656510       6.1409e-09      1.7532e-08           0.755          2.005  
   17     -84.38656511       9.0819e-09      6.5539e-09           0.755          2.005  

nuclear repulsion      9.1882584177
total electronic     -84.3865651138

final total energy   -75.1983066960

Geometry
-----------
 O       0.0000     0.0000     0.0000 
 H       0.0000    -1.4305     1.1093 
 H       0.0000     1.4305     1.1093 

dipole momemts (Debye)
----------------------
 α-spin           x= -0.0000 y=  0.0000 z=  1.7478         resultant   1.7477
 β-spin           x= -0.0000 y=  0.0000 z=  1.5752         resultant   1.5751
 α+β-spin         x= -0.0000 y=  0.0000 z=  3.3229         resultant   3.3229

Lowdin populations
--------------------
α spin      [4.4177 0.2912 0.2912]
β spin      [3.3379 0.3310 0.3310]

charge
α+β spin    [0.2444 0.3778 0.3778]        net = 1.00 

Mayer bond orders
------------------
       H1        H2 
O0 | 0.6636    0.6636    
H1 |           0.0044    

Valency
O0  = 1.33:   H1  = 0.67:   H2  = 0.67:   

PySCF check is  True
'''