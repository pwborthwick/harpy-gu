from __future__ import division
import numpy as np

periodic_table = ['','H','He',
                     'Li','Be','B', 'C',  'N', 'O', 'F',  'Ne',
                     'Na','Mg','Al','Si', 'P', 'S', 'Cl', 'Ar']
subshell_momenta = {'s' : [(0,0,0)],
                    'p' : [(1,0,0),(0,1,0),(0,0,1)],
                    'd' : [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
                    'f' : [[3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2], [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]]}

van_der_waals_radii = {'':0.00, 'H':1.20, 'He':1.40, 'Li':1.82, 'Be':1.53, 'B':1.92, 'C':1.70, 'N':1.55, 'O':1.52, 'F':1.47,
                       'Ne':1.54, 'Na':2.27, 'Mg':1.73, 'Al':1.84, 'Si':2.10, 'P':1.80, 'S':1.80, 'Cl':1.75, 'Ar':1.88}

def CONSTANTS(name):
    #NIST constants

    NIST = {'au->debye': 2.541746,
            'bohr->angstrom': 0.52917721092,
            'hartree->eV': 27.21138602,
            'eV[-1]->nm': 1239.841701,
            'Eh': 4.359744722207e-18,
            'c': 2.99792458e10,
            'planck': 6.62607015e-34,
            'au->femtosecond': 0.02418884254,
            'alpha': 0.0072973525,
            'em2->amu': 1822.8884850,
            'avogadro': 6.022140857e+23,
            'ke':8.854187817e-12,
            'bohr magneton' : 9.274009994e-24,
            'e': 1.6021766208e-19}

    return NIST[name]

class molecule(object):
    #molecule class - creates a molecule object
    '''
    properties - atom{symbol,number,center}
                 units   [string]
                 charge  [integer]
                 spin    [integer]
                 basis   [string]
                 orbital{atom,primatives,momenta,exponents,coefficients,normals}

                 natm, norb, nele
    methods    - nuclear_repulsion()
    '''
    def __init__(self, definition, units='angstrom', charge=0, spin=0, gto='sto-3g', gauge=[0,0,0], silent=False):

        class basis(object):

            def __init__(self, atom, primatives, momentum, exponents, coefficients, normals):

                self.atom = atom
                self.symbol = ''
                self.primatives = primatives
                self.momenta = np.array(momentum)
                self.exponents = np.array(exponents)
                self.coefficients = np.array(coefficients)
                self.normals = np.array(normals)

        class atoms(object):

            def __init__(self, id, symbol, center):

                self.id     = id
                self.symbol = symbol
                self.center = np.array(center, dtype=float)
                self.number = periodic_table.index(symbol)

        self.atom = []
        if type(definition) == list:
            for n, a in enumerate(definition):
                self.atom.append(atoms(n, a[0], a[1]))
        if type(definition) == str:
            self.build_from_internal_coordinates(atoms, definition)

        self.units = units
        if units == 'angstrom':
            for i in self.atom: i.center /=  0.52917721092

        self.gauge = gauge
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.name = gto

        self.orbital = []
        self.get_atom_basis('bas/' + self.name)

        self.natm = len(self.atom)
        self.norb = len(self.orbital)
        self.nele = sum([i.number for i in self.atom]) - charge

        self.silent = silent

    def nuclear_repulsion(self):
            #nuclear repulsion

        eNuc = 0.0
        natm = self.natm

        for i in range(natm):
            for j in range(i+1, natm):
                r = np.linalg.norm(self.atom[i].center - self.atom[j].center)
                eNuc += self.atom[i].number*self.atom[j].number / r

        return eNuc

    def charge_center(self):
        #center of nuclear charge

        molecular_charge = sum([i.number for i in self.atom])
        charge_center = [i.center[dim] * i.number/molecular_charge for i in self.atom for dim in range(3)]

        return [sum(charge_center[::3]), sum(charge_center[1::3]), sum(charge_center[2::3])]

    def get_atom_basis(self, path):
        #read atom blocks from BSE file

        def process_atom_block(block):
            #interpret BSE file data for atom (psi4 format)

            gto = {'m':[], 'p':[],'e':[],'c':[]}

            n = 0
            while True:

                m, p, _ = block[n].split()
                gto['m'].append(m.lower()) ; gto['p'].append(int(p))

                for i in range(gto['p'][-1]):
                    contractions = block[n+i+1].split()
                    gto['e'].append(float(contractions[0].replace('D','e')))
                    gto['c'].append(float(contractions[1].replace('D','e')))
                    if gto['m'][-1] == 'sp': gto['c'].append(float(contractions[2].replace('D','e')))

                n += gto['p'][-1] + 1

                if n >= len(block)-1: return gto

        def create_basis_table(type_, gto, atom_table):
                #create instances of basis class

            ke = 0 ; kc = 0
            for n,m in enumerate(gto['m']):

                if m != 'sp':
                    for L in subshell_momenta[m]:
                        atom_table.append(self.basis(type_, gto['p'][n], L, gto['e'][ke:ke+gto['p'][n]], gto['c'][ke:ke+gto['p'][n]], []))
                    ke += gto['p'][n] ; kc = ke

                else:
                    L = subshell_momenta['s'][0]
                    atom_table.append(self.basis(type_, gto['p'][n], L, gto['e'][ke:ke+gto['p'][n]], gto['c'][kc:kc+2*gto['p'][n]:2], []))
                    for L in subshell_momenta['p']:
                        atom_table.append(self.basis(type_, gto['p'][n], L, gto['e'][ke:ke+gto['p'][n]], gto['c'][kc+1:kc+2*gto['p'][n]:2], []))
                    ke += gto['p'][n] ; kc += 2*gto['p'][n]

            return atom_table

        def normalisation_factor(orb):
            #compute the normalisation for contracted Gaussian

            #double factorial
            from scipy.special import factorial2 as df

            #princpal quantum number (n-1)
            n = np.sum(orb.momenta)

            #double factorial terms
            norm = np.zeros((orb.primatives))

            double_factorial_term = df(2*orb.momenta[0]-1) * df(2*orb.momenta[1]-1) * df(2*orb.momenta[2]-1)
            for i,p in enumerate(orb.exponents):
                norm[i] = pow(2*p/np.pi,0.75)*pow(4*p,n/2) / np.sqrt(double_factorial_term)

            #coefficients normalisation
            pf = pow(np.pi,1.5) * double_factorial_term / pow(2,n)

            s = 0.0
            for i, p in enumerate(orb.exponents):
                for j, q in enumerate(orb.exponents):
                    s += norm[i] * norm[j] * orb.coefficients[i] * orb.coefficients[j] / pow(p + q , n + 1.5)

            s *= pf
            s = 1/np.sqrt(s)

            orb.coefficients  *= s

            return orb.coefficients, norm


        def create_basis(atom_table, symbol, n):
            #assign atom basis to molecular atoms

            for i in atom_table:

                if i.atom == symbol:
                    i.coefficients, normal = normalisation_factor(i)
                    self.orbital.append(self.basis(self.atom[n], i.primatives, i.momenta, i.exponents, i.coefficients, normal))

        #get atomic species in order of increasing atomic number
        species = list(set([i.symbol for i in self.atom])) ; order = [periodic_table.index(i) for i in species]
        order = sorted(order)
        species = [periodic_table[i] for i in order]

        atom_table = []

        #read down file processing element list until empty
        fp = open(path, 'r')
        while fp:

            line = fp.readline().strip()

            if line.lower() in ['cartesian','spherical']: type_ = line

            #get the file block for atom type
            if line[:2].strip() == species[0]:
                atom_basis = []
                while line != '****':
                    line = fp.readline().strip()
                    atom_basis.append(line)

                #decode the atom block
                gto = process_atom_block(atom_basis)

                #produce basis for atom types
                atom_table = create_basis_table(species[0], gto, atom_table)

                species.pop(0)
                if species == []: break

        fp.close()

        #construct basis
        for n,i in enumerate(self.atom):
            create_basis(atom_table, i.symbol, n)

    def build_from_internal_coordinates(self, atoms, definition):
        #build atom objects from internal coordinate form (z-matrix)

        def is_symbol(item):
            #determine if item can be interpreted as numeric

            try:
                float(item)
            except:
                return True
            else:
                return False

        def get_symbol_value(item):
            #symbolic value from symbol table

            value = [x.split('=') for x in subs if item in x]

            return float(eval(value[0][1]))

        def get_value(item, mode):
            #symbolic value and conversion

            value = get_symbol_value(item) if is_symbol(item) else item
            value = float(value) * np.pi/180.0 if mode in ['b', 't'] else float(value)

            return value

        def rodriguez(axis, theta):
            #rotation matrix generator

            axis /= np.linalg.norm(axis)
            psi = np.cos(theta/2)

            i, j ,k = -axis * np.sin(theta/2)

            return np.asarray([[psi*psi + i*i-j*j-k*k, 2*(i*j-psi*k), 2*(i*k+psi*j)],
                               [2*(i*j+psi*k), psi*psi + j*j-i*i-k*k, 2*(j*k-psi*i)],
                               [2*(i*k-psi*j), 2*(j*k+psi*i), psi*psi + k*k-i*i-j*j]])

        def process_atom(atom):
            #build atom definition from z-matrix line

            atom_data = geom[i].split()

            #special cases
            if atom >= 1:
                a = atom_data[1]
                stretch = get_value(atom_data[2], 's')
            if atom >= 2:
                b = atom_data[3]
                bend = get_value(atom_data[4], 'b')
            if atom >= 3:
                t = atom_data[5]
                dihedral = get_value(atom_data[6],'t')

            if atom == 0: return atom_data[0], [0.0, 0.0, 0.0]
            if atom == 1: return atom_data[0], [stretch, 0.0, 0.0]
            if atom == 2:
                #bend - angle between two vectors
                a, b = int(atom_data[1]), int(atom_data[3])
                u = coords[b-1, :] - coords[a-1, :]

                w = stretch*u/np.linalg.norm(u)
                w = np.dot(rodriguez([0, 0, 1], bend), w)

                return atom_data[0], w + coords[a-1, :]

            #general case
            if atom >= 3:
                #dihedral = angle between two planes
                a, b, c = int(atom_data[1]), int(atom_data[3]), int(atom_data[5])
                u = coords[b-1, :] - coords[a-1, :]
                v = coords[b-1, :] - coords[c-1, :]

                w = stretch*u/np.linalg.norm(u)
                normal = np.cross(u, v)
                w = np.dot(rodriguez(normal, bend), w)
                w = np.dot(rodriguez(u, dihedral), w)

                return atom_data[0], w + coords[a-1, :]

        #user input - split and process special characters
        specification = definition.split('\n')
        specification = [x.replace('\t', '    ') for x in specification if x != '']

        #differentiate geometry and symbol substitution table
        geom = [x for x in specification if not '=' in x]
        subs = [x for x in specification if '=' in x]

        natm = len(geom)

        #build the atom coordinate data and for non-dummy atoms create atom object
        coords = np.zeros((natm, 3))
        real_atom = 0
        for i in range(natm):
            z, coords[i, :] = process_atom(i)
            if z != 'X':
                self.atom.append(atoms(real_atom, z, coords[i, :]))
                real_atom += 1
