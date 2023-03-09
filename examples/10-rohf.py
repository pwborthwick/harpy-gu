from mol.mol import molecule
from scf.rohf import ROHF
import int.mo_spin as mos
from phf.eig import solver
from mol.mol import CONSTANTS

import numpy as np

mol = molecule([['O', ( 0.000000000000,  0.00000000000, 0.000000000000)], 
                ['H', ( 0.000000000000, -0.7570000000,  0.587000000000)], 
                ['H', ( 0.000000000000,  0.75700000000, 0.587000000000)]], 
                spin=1,
                units='angstrom',
                charge=-1,
                gto='sto-3g',
                silent=False)

scf = ROHF(mol, 
          cycles=50,
          tol=1e-10,
          diis=True)

scf_energy = scf.execute()

scf.analyse(method='dipole,geometry,charge,bonds')

'''
can do water with +1 charge - we can modify existing mol object if we're careful but probably safest to
create a new one. Here as we are changing the number of electrons we should reset the nele property
'''
scf.mol.charge = 1
scf.mol.nele = 9
scf = ROHF(mol, 
          cycles=50,
          tol=1e-10,
          diis=True)

scf_energy = scf.execute()

'''
******************
*   scf output   *
******************
method                  ROHF
charge                  -1
spin                    1 
units                   angstrom
open shell              True      :  multiplicity  2

basis is                sto-3g
analytic integration    aello cython - McMurchie-Davidson scheme

diis                    True  : buffer size is  6
scf control             maximum cycles are  50    :  convergence tolerance  1e-10

 cycle        E SCF              ΔE            |diis|               S²       multiplicity
------------------------------------------------------------------------------------------------------
    1     -82.30683343       8.2307e+01      7.2681e-01           0.750          2.000  
    2     -83.43254403       1.1257e+00      1.6752e+00           0.750          2.000  
    3     -83.43443895       1.8949e-03      7.5992e-02           0.750          2.000  
    4     -83.43457525       1.3630e-04      2.0502e-02           0.750          2.000  
    5     -83.43459035       1.5097e-05      6.2536e-03           0.750          2.000  
    6     -83.43459212       1.7648e-06      2.2726e-03           0.750          2.000  
    7     -83.43459292       8.0074e-07      1.9188e-03           0.750          2.000  
    8     -83.43459292       5.4001e-13      1.1136e-06           0.750          2.000  
    9     -83.43459292       2.8422e-14      1.9612e-10           0.750          2.000  

nuclear repulsion      9.1882584177
total electronic     -83.4345929164

final total energy   -74.2463344986

dipole momemts (Debye)
----------------------
 α-spin           x= -0.0000 y=  0.0000 z= -1.0636         resultant   1.0636
 β-spin           x= -0.0000 y= -0.0000 z=  0.3870         resultant   0.3870
 α+β-spin         x= -0.0000 y=  0.0000 z= -0.6766         resultant   0.6766

Geometry
-----------
 O       0.0000     0.0000     0.0000 
 H       0.0000    -1.4305     1.1093 
 H       0.0000     1.4305     1.1093 

Lowdin populations
--------------------
α spin      [4.4962 0.7519 0.7519]
β spin      [3.9732 0.5134 0.5134]

charge
α+β spin    [-0.4693 -0.2653 -0.2653]        net = -1.00

Mayer bond orders
------------------
       H1        H2 
O0 | 0.6197    0.6197    
H1 |           0.1025    

Valency
O0  = 1.24:   H1  = 0.72:   H2  = 0.72:

******************
*   scf output   *
******************
method                  ROHF
charge                  1 
spin                    1 
units                   angstrom
open shell              True      :  multiplicity  2

basis is                sto-3g
analytic integration    aello cython - McMurchie-Davidson scheme

diis                    True  : buffer size is  6
scf control             maximum cycles are  50    :  convergence tolerance  1e-10

 cycle        E SCF              ΔE            |diis|               S²       multiplicity
------------------------------------------------------------------------------------------------------
    1     -82.90989091       8.2910e+01      8.0755e-01           0.750          2.000  
    2     -83.84102519       9.3113e-01      1.8372e+00           0.750          2.000  
    3     -83.84212538       1.1002e-03      9.6334e-02           0.750          2.000  
    4     -83.84216522       3.9838e-05      1.2956e-02           0.750          2.000  
    5     -83.84216982       4.6023e-06      4.2071e-03           0.750          2.000  
    6     -83.84217051       6.9409e-07      1.5109e-03           0.750          2.000  
    7     -83.84217057       5.9718e-08      6.1713e-04           0.750          2.000  
    8     -83.84217057       6.3949e-13      1.9311e-06           0.750          2.000  
    9     -83.84217057       2.8422e-14      7.2992e-09           0.750          2.000  

nuclear repulsion      9.1882584177
total electronic     -83.8421705728

final total energy   -74.6539121550
'''

