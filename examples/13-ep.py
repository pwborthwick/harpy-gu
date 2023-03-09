from __future__ import division
from scf.rhf import RHF
from mol.mol import molecule
from phf.ept import EP
from mol.mol import CONSTANTS

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

    #NOTE: all minus signs gave been left in although for IP Koopman's theorem states the IP is the negative of the
    #orbital energy (and the sigmas)
    
    #create an instance of the electron propagator class - specify an orbital range in terms of frontier orbitals
    ep = EP(scf, orbital_range='HOMO-3,LUMO')

    #electron propagator order 2 - self-energy in diagonal approximation
    ep.method(order=2)

    print('\nElectron Propagator (2)\n-----------------------')
    print('orbital                Koopman                    EP2\n--------------------------------------------------------------')
    for orbital, energy in enumerate(ep.cache):

        #all orbitals are referred to base orbital 1, note orbital range property is requested range as tuple of orbitals based at 0
        homo , lumo = scf.mol.nele[0], scf.mol.nele[0] + 1
        spatial = orbital + 1 + ep.orbital_range[0]

        orbital_description = 'homo-' + str(homo - spatial) if spatial <= homo else 'lumo+' + str(spatial - lumo)
        orbital_description = orbital_description.replace('-0','').replace('+0','')

        #cache is list of dictionaries
        sigma, koopman = energy.values()
        eV =  CONSTANTS('hartree->eV')
        
        print(' {:10}   {:>10.4f}  {:>10.4f}    {:>10.4f}  {:>10.4f} '.
        format(orbital_description, koopman, koopman * eV, sigma, sigma * eV))

    #electron propagator order 3
    ep.method(order=3)

    print('\nElectron Propagator (3)\n-----------------------')
    print('orbital                Koopman                    EP3\n--------------------------------------------------------------')
    for orbital, energy in enumerate(ep.cache):

        #all orbitals are refered to base orbital 1
        homo , lumo = scf.mol.nele[0], scf.mol.nele[0] + 1
        spatial = orbital + 1 + ep.orbital_range[0]

        orbital_description = 'homo-' + str(homo - spatial) if spatial <= homo else 'lumo+' + str(spatial - lumo)
        orbital_description = orbital_description.replace('-0','').replace('+0','')

        sigma, koopman = energy.values()
        eV = CONSTANTS('hartree->eV')
        
        print(' {:10}   {:>10.4f}  {:>10.4f}    {:>10.4f}  {:>10.4f} '.
        format(orbital_description, koopman, koopman * eV, sigma, sigma * eV))

    #there is an 'approximate Green's function order 2 method which gives the components of the Green function correction to
    #to the orbital energy
    ep.approximate_greens_function_order_2()

    print('\nApproximate Green\'s Function (2)  (eV)\n--------------------------------------')
    print('orbital          Koopman       orbital relaxation    pair relaxation     pair removal     AGF(2) IP')
    print('-----------------------------------------------------------------------------------------------------')
    for orbital, energy in enumerate(ep.cache):
            
        #all orbitals are refered to base orbital 1
        homo , lumo = scf.mol.nele[0], scf.mol.nele[0] + 1
        spatial = orbital + 1 + ep.orbital_range[0]

        orbital_description = 'homo-' + str(homo - spatial) if spatial <= homo else 'lumo+' + str(spatial - lumo)
        orbital_description = orbital_description.replace('-0','').replace('+0','')

        prm, prx, orx, koo = energy.values()
        eV = CONSTANTS('hartree->eV')
        
        print(' {:10}    {:>10.4f}        {:>10.4f}           {:>10.4f}        {:>10.4f}      {:>10.4f}'.
        format(orbital_description, koo * eV, orx * eV, prx * eV, prm * eV, (koo + orx + prx + prm) * eV))

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

Electron Propagator (2)
-----------------------
orbital                Koopman                    EP2
--------------------------------------------------------------
 homo-3          -1.2097    -32.9175       -1.0974    -29.8609 
 homo-2          -0.5480    -14.9109       -0.5365    -14.5980 
 homo-1          -0.4365    -11.8785       -0.3800    -10.3404 
 homo            -0.3876    -10.5468       -0.2721     -7.4046 
 lumo             0.4776     12.9967        0.4739     12.8951 

Electron Propagator (3)
-----------------------
orbital                Koopman                    EP3
--------------------------------------------------------------
 homo-3          -1.2097    -32.9175       -1.0741    -29.2272 
 homo-2          -0.5480    -14.9109       -0.5452    -14.8363 
 homo-1          -0.4365    -11.8785       -0.3957    -10.7673 
 homo            -0.3876    -10.5468       -0.3007     -8.1827 
 lumo             0.4776     12.9967        0.4757     12.9445 

Approximate Green's Function (2)  (eV)
--------------------------------------
orbital          Koopman       orbital relaxation    pair relaxation     pair removal     AGF(2) IP
-----------------------------------------------------------------------------------------------------
 homo-3          -32.9175           10.8902              -5.7742           -0.2181        -28.0196
 homo-2          -14.9109            3.3608              -2.3881           -0.6473        -14.5855
 homo-1          -11.8785            8.2220              -6.1309           -0.4429        -10.2303
 homo            -10.5468           13.9522             -10.4463           -0.0276         -7.0685
'''