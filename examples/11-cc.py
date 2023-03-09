if __name__ == '__main__':
    import numpy as np

    from mol.mol import molecule, CONSTANTS
    from scf.rhf import RHF

    import int.mo_spin as mos
    from int.aello import aello

    from phf.cct import CC, rCC
    '''
    benchmarking against Hirata reference: https://hirata-lab.chemistry.illinois.edu/cc_data.out
    '''
    mol = molecule([['O', (  0.000000000000000,  0.000000000000000,  0.000000000000000)], 
                    ['H', (  0.000000000000000,  1.079252144093028,  1.474611055780858)], 
                    ['H', (  0.000000000000000,  1.079252144093028, -1.474611055780858)]], 
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

    '''
    create instance of CC object - pass scf object and boolean keyword for whether to use DIIS - default is True
    cc method has a cache property which is a listF for all methods in first section list contains one item the 
    cluster energy correction excepyfor ccsd(t) where a second item the (t) perturbative correction is also returned.
    In parallel create restricted-spin object for CCD and CCSD
    '''
    cc  =  CC(scf, diis=True)
    rcc = rCC(scf, diis=True)
    '''
    CCD method
    '''
    print('\n***************************\n*       CCD method        *\n***************************')
    cc.method(code='ccd', silent=True)
    print('CCD correction energy      {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))

    assert np.isclose(cc.cache[0], -0.0498521356)

    rcc.method('ccd', silent=True)
    print('CCD correction energy      {:<14.10f}'.format(rcc.cache[0]), '   spin-restricted')

    assert np.isclose(rcc.cache[0], -0.0498521356)

    '''
    CCSD method
    '''
    print('\n***************************\n*       CCSD method       *\n***************************')
    cc.method('ccsd', silent=True)
    print('CCSD correction energy     {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))
    
    assert np.isclose(cc.cache[0], -0.0501273286)

    rcc.method('ccsd', silent=True)
    print('CCSD correction energy     {:<14.10f}'.format(rcc.cache[0]), '   spin-restricted')

    assert np.isclose(rcc.cache[0], -0.0501273286)
    '''
    CCSD(T) method
    '''
    print('\n***************************\n*     CCSD(T) method      *\n***************************')
    cc.method('ccsd(t)', silent=True)
    print('CCSD correction energy     {:<14.10f}'.format(cc.cache[0]))
    print('(T) correction energy      {:<14.10f}'.format(cc.cache[1]))
    print('Corrected HF energy       {:<14.10f}'.
                                    format(cc.cache[0]+ cc.cache[1] + scf.reference[0] + scf.reference[1]))

    '''
    Linear CCD method
    '''
    print('\n***************************\n*       LCCD method       *\n***************************')
    cc.method('lccd', silent=True)
    print('LCCD correction energy     {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))

    assert  np.isclose(cc.cache[0], -0.0505753360)

    '''
    Linear CCSD method
    '''
    print('\n***************************\n*      LCCSD method       *\n***************************')
    cc.method('lccsd', silent=True)
    print('LCCSD correction energy    {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))

    assert  np.isclose(cc.cache[0], -0.0508915694)

    '''
    QCISD method
    '''
    print('\n***************************\n*      QCISD method       *\n***************************')
    cc.method('qcisd', silent=True)
    print('QCISD correction energy    {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))

    assert  np.isclose(cc.cache[0], -0.0501452655)

    '''
    lambda-CCSD
    '''
    print('\n***************************\n*     \u039B-CCSD method       *\n***************************')
    cc.ccsd('lambda', silent=True)
    print('\u039B-CCSD psuedo energy       {:<14.10f}'.format(cc.cache[0]))
    print('\u039B-CCSD Lagrange energy     {:<14.10f}'.format(cc.cache[1]))

    '''
    lambda-CCSD(t)
    '''
    print('\n****************************\n*    \u039B-CCSD(t) method      *\n****************************')
    cc.ccsd('lambda(t)', silent=True)
    print('\u039B-CCSD(t) psuedo energy    {:<14.10f}'.format(cc.cache[0]))
    print('\u039B-CCSD(t) Lagrange energy  {:<14.10f}'.format(cc.cache[1]))
    print('\u039B-CCSD(t) (t) energy       {:<14.10f}'.format(cc.cache[2]))

    '''
    one-particle reduced density matrix
    '''
    print('\n****************************\n*    one-particle rdm      *\n****************************')
    cc.ccsd('oprdm', silent=True)
    rdm = mos.spin_to_spatial(cc.cache, type='d')
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(rdm)

    #convert to AO basis
    mo_coeff = scf.get('c')
    ao_rdm = np.einsum('pi,ij,qj->pq', mo_coeff, rdm, mo_coeff, optimize=True)

    #get electronic dipole
    dipole = -np.array(aello(scf.mol.atom, scf.mol.orbital, 'dipole', None, scf.mol.charge_center())) 
    mu = np.einsum('xii->x',np.einsum('pr,xrq->xpq', ao_rdm, dipole, optimize=True))

    mu *= CONSTANTS('au->debye')
    print('\nCCSD electronic dipole  x={:>6.4f}  y={:>6.4f}  z={:>6.4f} Debye'.format(mu[0], mu[1], mu[2]))

    '''
    change model for benchmark : https://www.duo.uio.no/bitstream/handle/10852/42315/1/OtnorliMasterThesis.pdf
    Chapter 9.
    '''
    mol = molecule([['O', ( 0.000000000000,   0.00000000000 ,-0.009000000000)], 
                    ['H', ( 1.515263000000,   0.00000000000 ,-1.058898000000)], 
                    ['H', (-1.515263000000,   0.00000000000 ,-1.058898000000)]], 
                    spin=0,
                    units='bohr',
                    charge=0,
                    gto='dz',
                    silent=True)

    scf = RHF(mol, 
              cycles=50,
              tol=1e-10,
              diis=True)

    scf_energy = scf.execute()
    if not scf.converged: exit('SCF convergence failed')

    cc = CC(scf, diis=True)
    
    '''
    CCSDT-1a method
    '''
    print('\n***************************\n*     CCSDT-1a method     *\n***************************')
    cc.method('ccsdt-1a', silent=True)
    print('CCSDT-1a correction energy {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))

    assert np.isclose(cc.cache[0], -0.147577)

    '''
    CCSDT-1b method
    '''
    print('\n***************************\n*     CCSDT-1b method     *\n***************************')
    cc.method('ccsdt-1b', silent=True)
    print('CCSDT-1b correction energy {:<14.10f}'.format(cc.cache[0]))
    print('Corrected HF energy       {:<14.10f}'.format(cc.cache[0] + scf.reference[0] + scf.reference[1]))

    assert np.isclose(cc.cache[0], -0.147580)



'''
***************************
*       CCD method        *
***************************
CCD correction energy      -0.0498521365 
Corrected HF energy       -75.0125152226

***************************
*       CCSD method       *
***************************
CCSD correction energy     -0.0501273293 
Corrected HF energy       -75.0127904153

***************************
*     CCSD(T) method      *
***************************
CCSD correction energy     -0.0501273293 
(T) correction energy      -0.0014989534 
Corrected HF energy       -75.0142893687

***************************
*       LCCD method       *
***************************
LCCD correction energy     -0.0505753369 
Corrected HF energy       -75.0132384229

***************************
*      LCCSD method       *
***************************
LCCSD correction energy    -0.0508915703 
Corrected HF energy       -75.0135546563

***************************
*      QCISD method       *
***************************
QCISD correction energy    -0.0501452663 
Corrected HF energy       -75.0128083524

***************************
*     Λ-CCSD method       *
***************************
Λ-CCSD psuedo energy       -0.0493875286 
Λ-CCSD Lagrange energy     -0.0501273293 

****************************
*    Λ-CCSD(t) method      *
****************************
Λ-CCSD(t) psuedo energy    -0.0493875286 
Λ-CCSD(t) Lagrange energy  -0.0493875286 
Λ-CCSD(t) (t) energy       -0.0000757546 

****************************
*    one-particle rdm      *
****************************
[[ 2.      0.     -0.      0.     -0.      0.0001 -0.    ]
 [ 0.      1.9919  0.      0.0097 -0.      0.0009  0.    ]
 [-0.      0.      1.9735  0.      0.     -0.     -0.0036]
 [ 0.      0.0097  0.      1.9819  0.      0.0261  0.    ]
 [-0.     -0.      0.      0.      1.9984  0.     -0.    ]
 [ 0.0001  0.0009 -0.      0.0261  0.      0.0275  0.    ]
 [-0.      0.     -0.0036  0.     -0.      0.      0.0267]]

CCSD electronic dipole  x=-0.0000  y=1.5818  z=0.0000 Debye

***************************
*     CCSDT-1a method     *
***************************
CCSDT-1a correction energy -0.1475767866 
Corrected HF energy       -76.1574143893

***************************
*     CCSDT-1b method     *
***************************
CCSDT-1b correction energy -0.1475803316 
Corrected HF energy       -76.1574179344
'''
