from __future__ import division

import numpy as np
from itertools import combinations
from math import factorial, floor
import int.mo_spin as mos 

class Slater(object):
    '''
    class for Slater determinant bit-string treatment]
    '''

    def __init__(self, indexlist=None, determinant=0):
        '''
        instantiate Slater class object as either list of occupied orbitals or
        as a denary integer representation of binary string.
        '''
        if determinant == 0 and indexlist != None:
            determinant = Slater.indexlist_to_determinant(indexlist)
 
        self.determinant = determinant

    @staticmethod
    def reference():
        '''
        define the ground state determinant
        '''
        reference_state = [i for i in range(nocc)]
        
        return Slater(indexlist=reference_state)
    
    @staticmethod
    def count_set_bits(bits):
        '''
        return the number of occupations in integer
        '''
        return bin(bits).count('1')
    
    @staticmethod
    def determinant_to_indexlist(determinant):
        '''
        return the occupied orbital list from a bit integer representation
        '''
        return [n for n, bit in enumerate(bin(determinant)[2:][::-1] )if bit == '1']

    @staticmethod
    def indexlist_to_determinant(orbital_list):
        '''
        convert a list of occupied orbitals to a determinant integer
        '''
        bit = 0
        return sum([bit ^ (1 << i) for i in orbital_list])
        
    @staticmethod
    def unique_bits(p, q):
        '''
        return the common bits between two Slater determinant bit representations
        '''
        common = p & q

        return Slater.determinant_to_indexlist(p ^ common), Slater.determinant_to_indexlist(q ^ common)
    
    @staticmethod
    def orbital_positions_in_determinant(determinant, index_list):
        '''
        return the ordinal positions in orbital list of orbitals in Slater determinant
        '''        
        return [n for n, i in enumerate(Slater.determinant_to_indexlist(determinant)) if i in index_list]
        
    def common_orbitals_in_spin_scattered_indexlist(self, compare_with):
        '''
        return the common orbitals between two spin-scattered index lists
        '''
        return Slater.determinant_to_indexlist(self.determinant & compare_with.determinant) 

    
    def orbital_positions_in_indexlist(self, index_list):
        '''
        return the ordinal positions of indices in index lists
        '''
        return Slater.orbital_positions_in_determinant(self.determinant, index_list)

    def count_different_orbitals(self, compare_with):
        '''
        return the count of different orbitals between determinants
        '''
        return Slater.count_set_bits(self.determinant ^ compare_with.determinant) >> 1
    
    def permutation_parity_to_bit_zero(self, index_list):
        '''
        parity of permutations needed to move index lists to 0 bit position
        '''
        sign = 1
        position = self.orbital_positions_in_indexlist(index_list)

        for n, orbital in enumerate(position):
            if ((orbital - n) % 2 ) == 1:  sign *= -1

        return sign

    def unique_bits_and_parity(self, compare_with):
        '''
        return the common orbital lists between the two determinants and the parity of maximum coincidence
        '''
        bit_list = Slater.unique_bits(self.determinant, compare_with.determinant)

        sign = [        self.permutation_parity_to_bit_zero(bit_list[0]), 
                compare_with.permutation_parity_to_bit_zero(bit_list[1])]

        return (bit_list[0], bit_list[1]), sign[0] * sign[1]

    def determinant_spin_excitations(self):
        '''
        get the alpha and beta excitation numbers from determinant
        '''
        occupations = [(int(bool(self.determinant & (1 << 2*i))), int(bool(self.determinant & (2 << 2*i)))) for i in range(floor(nmo*0.5))]

        return [i.count(0) for i in zip(*occupations)]
    
    def unoccupied_orbital_positions_in_indexlist(self):
        '''
        return the ordinal positions of unoccupied orbitals list
        '''
        return [i for i in range(nmo) if self.determinant & (1 << i) == 0]
    
    def single_substitution(self, i, a):
        '''
        set orbital i to zero and orbital a to one in orbital bits
        '''
        det = self.clone()
        det.determinant =  (det.determinant & ~(1 << i)) | (1 << a)

        return det

    def singles_residue(self):
        '''
        construct all the single residues generated by Slater determinant
        '''
        occupied   = Slater.determinant_to_indexlist(self.determinant)
        unoccupied = self.unoccupied_orbital_positions_in_indexlist()

        determinant_list =  [self.single_substitution(i, a) for a in unoccupied 
                                                            for i in occupied]
        return determinant_list

    def double_substitution(self, i, a, j, b):
        '''
        set orbitals i,j to zero and orbitals a,b to one in orbital bits
        '''
        det = self.clone()
        det.determinant = (det.determinant & ~(1 << i)) | (1 << a)
        det.determinant = (det.determinant & ~(1 << j)) | (1 << b)

        return det

    def doubles_residue(self):
        '''
        construct all the double residues generated by Slater determinant
        '''
        occupied   = Slater.determinant_to_indexlist(self.determinant)
        unoccupied = self.unoccupied_orbital_positions_in_indexlist()

        determinant_list = [self.double_substitution(i, a, j, b) for a, b in combinations(unoccupied, 2) 
                                                                  for i, j in combinations(occupied, 2)]

        return determinant_list

    def clone(self):
        '''
        return a clone (deep copy) of a Slater object
        '''
        return Slater(determinant=self.determinant)

    def __str__(self):
        '''
        formatted representation of Slater determinant
        '''
        determinant_string = bin(self.determinant)[2:].zfill(nmo)

        return ('| (\u03B1) ' + determinant_string[1::2] 
              + '  (\u03B2) ' + determinant_string[::2] + '>')
        
                         
class Hamiltonian(object):
    '''
    class for Hamiltonian evalution by Slater-Condon rules
    '''
    def __init__(self, h_core_spin, g_spin):
        '''
        instantiate Hamiltonian class object with molecular spin basis core Hamiltonian and
        2-electron repulsion integrals
        '''
        self.h = h_core_spin
        self.g = g_spin

    def spin_adapt_determinants(determinant_list):
        '''
        removing determinants in which there is an a->b or b->a excitation
        '''
        for p in determinant_list[:]:
            nalpha, nbeta = p.determinant_spin_excitations()
 
            if nalpha != nbeta:
                determinant_list.remove(p)

        return determinant_list
            
    def build_Hamiltonian_matrix(self, determinant_list):
        '''
        generate the CI matrix
        '''
        dimension = len(determinant_list)
        matrix = np.empty((dimension, dimension))

        for i, p in enumerate(determinant_list):
            for j, q in enumerate(determinant_list[:(i+1)]):
                matrix[j, i] = matrix[i, j] = self.evaluate_matrix_element(p, q)
 
        return matrix
    
    def evaluate_matrix_element(self, p, q):
        '''
        Slater-Condon rules to evaluate matrix element formed by excitation between two
        Slater determinants
        '''
        excitation_degree = p.count_different_orbitals(q)

        if excitation_degree > 2: return 0.0

        if   excitation_degree == 0: return self.evaluate_zero_excitation(p)
        elif excitation_degree == 1: return self.evaluate_single_excitation(p, q)
        elif excitation_degree == 2: return self.evaluate_double_excitation(p, q)
        
    def evaluate_zero_excitation(self, p):
        '''
        determinants have no excitation between them, they are identical
        '''
        orbital_list = Slater.determinant_to_indexlist(p.determinant)
        
        value = 0.0
        for m in orbital_list:
            value += self.h[m, m]
        
        for i, m in enumerate(orbital_list[:-1]):
            for n in orbital_list[i:]:
                value += self.g[m, n, m, n]

        return value

    def evaluate_single_excitation(self, p, q):
        '''
        evaluate a matrix element formed by determinants with a single excitation
        '''
        (cp, cq), parity = p.unique_bits_and_parity(q)
        common = p.common_orbitals_in_spin_scattered_indexlist(q)

        m , p = cp[0], cq[0]

        value = self.h[m, p]

        for n in common:
            value += self.g[m, n, p, n]

        return parity * value 

    def evaluate_double_excitation(self, p, q):
        '''
        evaluate a matrix element formed by determinants with a double excitation
        '''
        (cp, cq), parity = p.unique_bits_and_parity(q)
        
        return parity * self.g[cp[0], cp[1], cq[0], cq[1]]

class FCI(object):
    '''
    Determinant Based Full Configuration Interaction class
    '''
    def __init__(self, scf, method='S', spin_adapt=False, use_residues=''):
        '''
        instantiate the DB-FCI class
        '''
        self.scf, self.method, self.spin_adapt, self.use_residues = scf, method, spin_adapt, use_residues

        global nmo, nocc
        nmo, nocc = scf.mol.norb * 2, sum(scf.mol.nele)

        if not method in ['S', 'D', 'T', 'Q', 'P', 'F']: exit('method [ ' + method + ' ] not recognized')
        
        full = min(nmo - nocc, nocc)
        self.degree = {'S':1, 'D':2, 'T':3, 'Q':4, 'P':5, 'F':full}[method]

        #sanity checks 
        if self.degree > nocc:
            print('**warn - cannot excite greater than number of electrons')
            self.degree = nocc
        if self.degree > (nmo - nocc):
            print('**warn - cannot excite greater than number of virtual orbitals')
            self.degree = (nmo - nocc)

        if self.use_residues != '':
            self.residues()
            return
        
        #construct the set of determinants generating the Hamiltonian
        determinant_list = [] if method == 'S' else [Slater.reference()]
        
        for degree in range(self.degree):
            determinant_list += self.determinant_basis(degree + 1)

        #reference state molecular spin quantities
        Hp, g = self.mo_spin_integrals()
        
        #Hamiltonian generation
        hamiltonian_generator = Hamiltonian(Hp, g)
        if self.spin_adapt:
            determinant_list = Hamiltonian.spin_adapt_determinants(determinant_list)

        self.hamiltonian = hamiltonian_generator.build_Hamiltonian_matrix(determinant_list)
        self.determinant_list = determinant_list
        
        #is_singlet for spin_adapted determinant basis only
        if self.spin_adapt:
            self.is_singlet == None

    #for spin-adapted determine if root is singlet
    def is_singlet(self, eigenvector):
        '''
        determine if eigenvector represents a singlet
        '''
        return np.isclose(sum(eigenvector), 0.0)

    def basis_dimension(self):
        '''
        calculate size of the determinant basis (Hamiltonian size)
        '''
        return int(factorial(nmo)/(factorial(nocc) * factorial(nmo - nocc)))
        
    def determinant_basis(self, degree):
        '''
        generate the list of determinants to be used to construct Hamiltonian elements
        '''
        #hole states
        holes     = [Slater.indexlist_to_determinant(i) for i in
                     list(combinations(range(nocc), nocc - degree))]
        #particle state
        particles = [Slater.indexlist_to_determinant(i) for i in
                     list(combinations(range(nocc, nmo),   degree))]

        basis = []
        for h in holes:
            for p in particles:

                basis.append(Slater(determinant=(h ^ p)))
 
        return basis

    def mo_spin_integrals(self):
        '''
        use mo_spin modules to convert ground state core Hamiltonian and 2-electron
        repulsion integrals to a molecular spin basis
        '''

        #g - the 2-electron repulsion integrals in MO spin basis
        tensor = mos.orbital_transform(self.scf, 'm', self.scf.get('i'))

        spin_block = np.kron(np.kron(tensor, np.eye(2)).transpose(), np.eye(2))
        g = spin_block.transpose(0,2,1,3) - spin_block.transpose(0,2,3,1)

        #Hp - one electron operator  in MO spin basis
        Hp = np.kron(mos.orbital_transform(self.scf, 'm', self.scf.get('t') + self.scf.get('v')), np.eye(2))

        return Hp, g

    def residues(self):
        '''
        compute CIS or CISD using residue lists
        '''
        #program execution control
        no_execute = ('*' in self.use_residues)
        self.use_residues = self.use_residues.replace('*','')
        if not self.use_residues in ['S', 'D', 'SD']: exit('CIS, CID or CISD only for residues method')
        
        reference_slater = Slater.reference()

        #build determinant lists for 's', 'D' or 'SD' methods
        determinant_list = [reference_slater] if self.use_residues in ['D', 'SD'] else []
        if self.use_residues in ['S','SD']:
            determinant_list += reference_slater.singles_residue()
        if self.use_residues in ['SD','D']:
            determinant_list += reference_slater.doubles_residue()

        #'*' indicates no execution for obtaining residues list only
        if no_execute:
            self.determinant_list = determinant_list
            return

        #instatiate a Hamiltonian object
        Hp, g = self.mo_spin_integrals()
        hamiltonian_generator = Hamiltonian(Hp, g)
   
        if self.spin_adapt:
            determinant_list = Hamiltonian.spin_adapt_determinants(determinant_list)
        
        self.hamiltonian = hamiltonian_generator.build_Hamiltonian_matrix(determinant_list)            
        self.determinant_list = determinant_list
        
    def __str__(self):
        '''
        formatted information
        '''
        information = (self.method + ',' + ''.join(['S','D','T','Q','P'][:self.degree]) + ','
                       + str(self.basis_dimension()) + ',' + str(self.hamiltonian.shape[0]) + ','
                       + ['No','Yes'][self.spin_adapt] + ',' + self.use_residues)
        
        return information