from __future__ import division
import numpy as np
from int.aello import aello
from mol.mol import subshell_momenta

def out(silent, data, key):
    #print data to console

    symbol = lambda i: ['s','px','py','pz'][[[0,0,0],[1,0,0],[0,1,0],[0,0,1]].index(i)]

    if not silent and key == 'initial':

        mol = data[0]
        print()
        print('******************\n*   scf output   *\n******************')

        print('method                  {:<4s}'.format(data[4]))
        print('charge                  {:<2d}'.format(mol.charge))
        print('spin                    {:<2d}'.format(mol.spin))
        print('units                  ',mol.units)
        print('open shell             ',data[3], '     :  multiplicity ', int(2*(mol.spin/2)+1))

        print('\nbasis is               ', mol.name)
        print('analytic integration    aello cython - McMurchie-Davidson scheme')
        print('\ndiis                   ', data[1],' : buffer size is ', data[2])

        if data[4] in ['RKS']:
            print('\nnumerical integration   Mura-Knowles, Lebedev')
            print('                        radial prune is Aldrich-Treutler')
            print('                        Becke partitioning scheme is Stratmann')
            print('                        radial adjustment is Treutler')
            print('                        integration order  period 1 (10,11)   period 2 (15,15)')
            print('mesh                   ', data[6])
            print('\nfunctional             ', data[5])


    if not silent and key == 'cycle':
        print('scf control             maximum cycles are ', data[0], '   :  convergence tolerance ', data[1])

    if not silent and key == 'rhf':
        cycle, eSCF, delta, norm, homo, lumo, warn = data
        if cycle == 1:
            print('\n cycle        E SCF              \u0394E            |diis|               homo            lumo')
            print('------------------------------------------------------------------------------------------------')
        if norm != '':
            print('   {:>2d}    {:>13.8f}     {:>12.4e}    {:>12.4e}        {:>12.8f}    {:>12.8f}   '.
                format(cycle, eSCF, delta, norm, homo, lumo))
        else:
            print('   {:>2d}    {:>13.8f}     {:>12.8f}                     {:>12.8f}    {:>12.8f}   '.
                format(cycle, eSCF, delta, homo, lumo), end='')
            print(warn)

    if not silent and key == 'uhf':
        cycle, eSCF, delta, norm, spin, warn = data
        ss, multiplicity = spin
        if cycle == 1:
            print('\n cycle        E SCF              \u0394E            |diis|               S\u00B2       multiplicity')
            print('------------------------------------------------------------------------------------------------------')
        if norm != '':
            print('   {:>2d}    {:>13.8f}     {:>12.4e}    {:>12.4e}          {:>6.3f}         {:>6.3f}  '.
                format(cycle, eSCF, delta, norm, ss, multiplicity))
        else:
            print('   {:>2d}    {:>13.8f}     {:>12.8f}                       {:>6.3f}         {:>6.3f}  '.
                format(cycle, eSCF, delta, ss, multiplicity), end='')
            print(warn)

    if not silent and key == 'rks':
        cycle, one_e, j, k, nele, delta, norm, warn = data
        if cycle == 1:
            print('\n cycle     1 electron         coulomb         exchange          electrons')
            print('                                                                                   \u0394E         diis norm')
            print('-------------------------------------------------------------------------------------------------------------')
        print('   {:>2d}     {:>12.8f}    {:>12.8f}    {:>12.8f}        {:>8.4f} '.format(cycle, one_e, j, k, nele))
        if norm != '':
            print('                                                                             {:>12.4e}  {:>12.4e} '.format(delta, norm))
        else:
            print('                                                                             {:>12.4e}            '.format(delta), end='')
            print(warn)

    if not silent and key == 'uks':
        if len(data) == 6:
            cycle, e1e, ej, ex, ne, et = data
            if cycle == 0:
                print('\n cycle     1 electron         coulomb         exchange          electrons           total')
                print('                                                                    S\u00B2      multiplicity            \u0394E         diis norm')
                print('----------------------------------------------------------------------------------------------------------------------------')
            print('   {:>2d}    {:>13.8f}     {:>12.8f}    {:>12.8f}    ({:>8.4f},{:>8.4f})  {:>12.8f} '.format(cycle+1, e1e, ej, ex, ne[0], ne[1], et))
        if len(data) == 3:
            spin_statistics, norm, delta = data
            ss, multiplicity = spin_statistics

            if type(norm) != str:
                print(' '*60, '     {:>6.3f}       {:>6.3f}         {:>12.6f}  {:>12.6f} '.format(ss, multiplicity, delta, norm))
            else:
                print(' '*91, ' {:>12.6f}            '.format(delta))

    if not silent and key == 'final':
        print('\nnuclear repulsion   {:>15.10f}'.format(data[1]))
        print('total electronic    {:>15.10f}'.format(data[0]))
        print('\nfinal total energy  {:>15.10f}'.format(data[0]+data[1]))

    if key == 'final-ks':
        print('\nfinal energies (Hartree)\n------------------------\none electron        {:>15.10f}'.format(data[1]))
        print('coulomb             {:>15.10f}'.format(data[2]))
        print('exchange            {:>15.10f}'.format(data[3]))
        print('nuclear repulsion   {:>15.10f}'.format(data[4]))
        print('total electronic    {:>15.10f}'.format(data[0]))
        print('\nfinal total energy  {:>15.10f}'.format(data[0]+data[4]))

    if not silent and key == 'rcharge':
        #molecular orbitals for alpha and beta spins

        p, q = data

        #Lowdin population analysis
        print('\nLowdin populations\n--------------------')

        print(' ',np.array2string(p, floatmode='maxprec_equal', max_line_width=80, precision=4))

        print('\ncharge')
        print(' ',np.array2string(q, floatmode='maxprec_equal', max_line_width=80, precision=4), end='')
        print('        net = {:<5.2f}\n'.format(np.sum(q)))

    if not silent and key == 'rdipole':

        mu = data
        print('\ndipole momemts (Debye)\n----------------------')
        print(' x={:>8.4f} y={:>8.4f} z={:>8.4f}         resultant {:>8.4f}'.format(mu[0], mu[1], mu[2], np.linalg.norm(mu)))

    if not silent and key == 'udipole':
        spin = ['\u03B1', '\u03B2']
        a, b = data[0,:], data[1,:]

        print('\ndipole momemts (Debye)\n----------------------')
        print(' {:1}-spin           x={:>8.4f} y={:>8.4f} z={:>8.4f}         resultant {:>8.4f}'
        .format(spin[0], a[0], a[1], a[2], np.linalg.norm(a)))
        print(' {:1}-spin           x={:>8.4f} y={:>8.4f} z={:>8.4f}         resultant {:>8.4f}'
        .format(spin[1], b[0], b[1], b[2], np.linalg.norm(b)))
        mu = a + b
        print(' {:1}+{:1}-spin         x={:>8.4f} y={:>8.4f} z={:>8.4f}         resultant {:>8.4f}'
        .format(spin[0], spin[1], mu[0], mu[1], mu[2], np.linalg.norm(mu)))

    if not silent and key == 'ucharge':
        #molecular orbitals for alpha and beta spins

        spin = ['\u03B1', '\u03B2']
        p, q = data

        #Lowdin population analysis
        print('\nLowdin populations\n--------------------')

        print(spin[0]+' spin     ',np.array2string(p[0], floatmode='maxprec_equal', max_line_width=80, precision=4))
        print(spin[1]+' spin     ',np.array2string(p[1], floatmode='maxprec_equal', max_line_width=80, precision=4))

        print('\ncharge')

        print(spin[0]+'+'+spin[1]+' spin   ',np.array2string(q, floatmode='maxprec_equal', max_line_width=80, precision=4), end='')
        print('        net = {:<5.2f}\n'.format(np.sum(q)))

    if key == 'geometry':

        mol = data
        print('\nGeometry\n-----------')
        for a in mol.atom:
            print('{:>2s}   {:>10.4f} {:>10.4f} {:>10.4f} '.format(a.symbol, a.center[0], a.center[1], a.center[2]))

    if key == 'bonds':
        #Mayer bond orders

        bond_order, valence, mol = data

        print('\nMayer bond orders\n------------------')
        for i in range(1, mol.natm): print('       {:3s}'.format(mol.atom[i].symbol + str(mol.atom[i].id)), end='')
        print()
        for i in range(mol.natm-1):
            print(mol.atom[i].symbol + str(mol.atom[i].id), '|','          '*i, end='')
            for j in range(i+1, mol.natm):
                print('{:<8.4f}  '.format(bond_order[i,j]), end='')
            print()

        print('\nValency')
        for i in range(mol.natm): print('{:4s}= {:<3.2f}:   '.format(mol.atom[i].symbol + str(mol.atom[i].id), valence[i]), end='')
        print()

    if key == 'orbitals':

        mol = data[0]
        if not silent:
            print('\nbasis analysis\n--------------')
            print('shell   l    pGTO    cGTO\n-------------------------')

        shells = 0; contracted = 0; primatives = 0
        basis_analysis = []

        for b in mol.orbital:
            basis_analysis.append([b.atom.id, np.sum(b.momenta),len(b.exponents), b.exponents[0]])

        nl = lambda x: (x+1)*(x+2)//2

        symbol = {0:'s',1:'p',2:'d',3:'f'}

        mol_id = 0
        #group by atom center
        for a in mol.atom:
            atom_basis = [i for i in basis_analysis if i[0] == a.id]
            if not silent: print('{:<2s}   {:<2}'.format(a.symbol, a.id))

            #group by momentum
            shell = [1, 2, 3, 4]
            for l in range(3):
                momentum_basis = [i for i in atom_basis if i[1] == l]

                #polarized have multiple centers
                expected_contraction = len(momentum_basis)//nl(l)
                shell_basis = list(set(tuple(x) for x in momentum_basis))
                actual_contraction = len(shell_basis)

                contracted += len(shell_basis) * nl(l)

                #adjust contacted count for multiple center
                delta_centers = expected_contraction - actual_contraction
                contracted += delta_centers * nl(l)

                for s in shell_basis:
                    centers = int([i[3] for i in momentum_basis].count(s[3])/nl(l))
                    if not silent: print('  {:<2d}    {:<2d}     {:<2d}      {:<1d}'.format(shells,s[1],s[2], centers))
                    shells += 1
                    primatives += s[2]*nl(l)

                    #construct the [n][l][m] designation of orbital
                    momentum = subshell_momenta[symbol[l]]
                    for center in range(centers):
                        for n in range(nl(l)):
                            i,j,k = momentum[n]
                            mol.orbital[mol_id].symbol = str(shell[l])+symbol[l]+'x'*i+'y'*j+'z'*k
                            mol_id += 1

                        shell[l] += 1
        if not silent:
            print('\n-------------------------')
            print('number of shells {:>12d}\nnumber of primative GTO {:>5d}\nnumber of contracted GTO {:>4d}'.format(shells, primatives, contracted))


