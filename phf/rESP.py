from __future__ import division
import numpy as np

import pylab as p
from mol.mol import CONSTANTS, van_der_waals_radii
from mol.utl import is_bond, thomson_distribution
from int.aello import aello

import os

'''
A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the RESP model. Christopher I. Bayly
- C. I. Bayly et. al. J. Phys. Chem. 97, 10269 (1993)
Connolly algorithm - [https://en.wikipedia.org/wiki/Accessible_surface_area]
Fibonacci Algorithm - [https://code-examples.net/en/q/927f21]
Saff and Kuijlaars algorithm - [https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf]
Thomson problem -[https://en.wikipedia.org/wiki/Thomson_problem]
'''

class RESP(object):
    #restrained electrostatic potential class

    def __init__(self, scf):

        self.scf = scf
        self.parameters = {}

        self.set_defaults()

    def file_write(self, inverse_radii):
        #save core data to file for re-start

        np.savez('rESP', rad= self.surface,
                       pot= self.electrostatic_potential,
                       inv= inverse_radii,
                       par= self.parameters)

    def file_read(self):
        #read core data for re-start

        data = np.load('rESP.npz', allow_pickle = 'TRUE')

        self.electrostatic_potential = data['pot']
        self.surface = data['rad']

        if 'c' in self.parameters['file']: self.file_clear()
        self.parameters = data['par'].item()
        self.parameters['file'] = ''

        return data['inv']

    def file_clear(self):
        #remove the saved data file

        os.remove('rESP.npz')

    def execute(self):
        #compute the rESP

        self.cache = {}

        #handle the file requests
        if not 'r' in self.parameters['file']:
            self.get_radii()
            self.generate_surface_distributions()
            self.generate_potential_field()
            inverse_radii = self.get_inverse_radii()
        else:
            inverse_radii = self.file_read()

        if self.parameters['file'] == 'w': self.file_write(inverse_radii)

        if self.parameters['constrain'] is not None: self.build_constraints()

        #generate system of equations
        matrix = self.build_matrix(inverse_radii)
        vector = self.build_vector(inverse_radii)

        #generate the constrained charges
        matrix, vector = self.generate_constrained_charge(matrix, vector)
        self.solve_constrained_charges(matrix, vector)

        key = 'constrained' if self.parameters['constrain'] is not None else 'free'
        self.cache[key] = self.constrained_charges

        #generate restrained charges if selected
        if self.parameters['restrain'] is not None or self.parameters['refine']:
            self.cache['cycles'] = self.solve_restrained_charges(matrix, vector, self.constrained_charges[:])
            self.cache['restrained'] = self.restrained_charges

        self.get_carbon_groups()

        #stage 2 refinements
        if self.parameters['refine'] and (self.carbon_groups != []):

            self.refine_charges()
            self.build_constraints()

            matrix = self.build_matrix(inverse_radii)
            vector = self.build_vector(inverse_radii)

            matrix, vector = self.generate_constrained_charge(matrix, vector)

            self.solve_constrained_charges(matrix, vector)

            if 'a' in self.parameters['refine']: self.parameters['restrain']['a'] = self.parameters['refine']['a']
            if 'b' in self.parameters['refine']: self.parameters['restrain']['b'] = self.parameters['refine']['b']

            self.solve_restrained_charges(matrix, vector, self.constrained_charges[:])
            self.cache['refined constrained'] = self.constrained_charges
            self.cache['refined restrained'] = self.restrained_charges

        #root mean square deltas with classical computation
        self.cache['root mean square'] = {}
        self.cache['root mean square'][key]          = self.classical_charges(self.cache[key], inverse_radii)
        if self.parameters['restrain'] is not None:
            self.cache['root mean square']['restrained'] = self.classical_charges(self.cache['restrained'], inverse_radii)

        #check charges sum to net charge on molecule
        self.assert_charges()

    def set_defaults(self):
        #set the default parameters

        shell = {}
        shell['distribution'] = 'connolly'
        shell['density'] = 1
        shell['count'] = 1
        shell['scale'] = 1.0
        shell['radii'] = None
        self.parameters['shell'] = shell

        self.constraints_count = 0
        self.parameters['constrain'] = None
        self.parameters['restrain'] = None

        self.parameters['file'] = ''

        #do a second stage refinement C-H on same carbon made equal
        self.parameters['refine'] = False

    def set_parameters(self, **kwargs):
        #set class property parameters

        for key in kwargs.keys():
            if key == 'radii':
                self.parameters['shell']['radii'] = kwargs[key]
                self.get_radii()
            if key in ['distribution', 'density', 'count', 'scale', 'view']:
                self.parameters['shell'][key] = kwargs[key]
            if key == 'points':
                self.parameters['shell']['points'] = kwargs[key]
                del self.parameters['shell']['density']
            if key == 'increment' and self.parameters['shell']['count'] > 1:
                self.parameters['shell']['increment'] = kwargs[key]
            if key == 'distribution':
                self.parameters['shell']['distribution'] = kwargs[key]

            if key == 'constrain':
                self.parameters['constrain'] = []
                for constraint in kwargs[key]:
                    self.parameters['constrain'].append(constraint)

            if (self.parameters['shell']['count'] > 1) and ('increment' not in self.parameters['shell']):
                self.parameters['shell']['increment'] = 0.2

            if key == 'restrain':
                self.parameters['restrain'] = {'a': 5e-4, 'b': 1e-1, 'H': True,
                                               'tol':self.scf.tol, 'cycles':self.scf.cycles}
                for restraint in kwargs[key]:
                    self.parameters['restrain'][restraint] = kwargs[key][restraint]

            if key == 'refine':
                self.parameters['refine'] = {'apply': True, 'a': 1e-3, 'b': 1e-1}
                for refine in kwargs[key]:
                    self.parameters['refine'][refine] = kwargs[key][refine]

            if key == 'file':
                self.parameters['file'] = kwargs[key]
                if kwargs[key] == 'r' and not os.path.isfile('rESP.npz'):
                    self.parameters['file'] = 'w'

    def get_radii(self):
        #set class property radii to Van de Waals radii - units Angstrom

        vdw = van_der_waals_radii.copy()

        #override default Van de Waals radii
        if self.parameters['shell']['radii'] is not None:
            for x in self.parameters['shell']['radii']:
                vdw[x[0]] = x[1]

        self.radii = np.array([vdw[x.symbol] for x in self.scf.mol.atom])

    def surface_distribution(self, shell_points, shell, atom):
        #distribute specified points over spherical surface

        surface_points = np.zeros((shell_points, 3))

        #A Connolly distribution algorithm
        if self.parameters['shell']['distribution'] == 'connolly':

            point_count = [int(np.sqrt(np.pi * shell_points)), 0]
            point_count[1] = point_count[0]//2

            nu = 0
            for i in range(point_count[1] + 1):

                phi = np.pi * i / point_count[1]
                z, xy = np.cos(phi), np.sin(phi)

                nh = max(int(point_count[0] * xy + 1e-10), 1)

                for j in range(nh):

                    psi = 2*np.pi * j / nh
                    x, y = np.cos(psi)*xy, np.sin(psi)*xy

                    if nu >= shell_points: break

                    surface_points[nu, :] = [x, y, z]
                    nu += 1

            #remove zero rows
            surface_points = np.delete(surface_points, slice(nu, None), axis=0)

        #Fibonacci sphere algorithm
        if self.parameters['shell']['distribution'] == 'fibonacci':

            dtheta, dz = np.pi*(3.0 - np.sqrt(5.0)),  2.0/shell_points
            theta, z   =  0.0, 1.0 - dz * 0.5

            for p in range(shell_points):

                r = np.sqrt(1.0 - z*z)
                surface_points[p, :] = [np.cos(theta)*r, np.sin(theta)*r, z]

                #update positions
                z -= dz
                theta += dtheta

        if self.parameters['shell']['distribution'] == 'saff and kuijlaars':
            s, dz = 3.6/np.sqrt(shell_points),  2.0/shell_points
            theta, z   =  0.0, 1.0 - dz * 0.5

            for p in range(shell_points):

                r = np.sqrt(1.0 - z*z)
                surface_points[p, :] = [np.cos(theta)*r, np.sin(theta)*r, z]

                #update positions
                z -= dz
                theta += s/r

        if self.parameters['shell']['distribution'] == 'thomson':
            surface_points, cycle, u_thomson = thomson_distribution(shell_points, view=None)
            self.cache['thomson'].append({'atom':atom, 'shell': shell, 'cycles':cycle, 'U':u_thomson,
                                          'points':shell_points})

        return surface_points

    def generate_surface_distributions(self):
        #construct the ensemble of points on VdW surfaces of atom shells - units Angstrom

        global_surface = None
        self.atom_index = np.zeros(self.scf.mol.natm)

        external = lambda x, y: (np.linalg.norm(surface[x, :] - self.scf.mol.atom[y].center[:] * CONSTANTS('bohr->angstrom'))
                                 >= (self.radii[y] * scaling_factor))

        #if Thomson distribution define cache list for details
        if self.parameters['shell']['distribution'] == 'thomson':
            self.cache['thomson'] = []

        #fixed number of points
        if 'points' in self.parameters['shell']:
            generate = self.parameters['shell']['points']

        #for each shell
        for shell in range(self.parameters['shell']['count']):

            #scaling factor for shell VdW radius
            shell_increment = self.parameters['shell']['increment'] if 'increment' in self.parameters['shell'] else 0
            scaling_factor = self.parameters['shell']['scale'] + (shell * shell_increment)

            for atom in range(self.scf.mol.natm):

                #effective shell radius accounting for scale factor
                shell_radius = self.radii[atom] * scaling_factor

                #if density depends on radius ie density- recalculate for scaling
                if 'density' in self.parameters['shell']:
                    generate = int(4.0 * np.pi * shell_radius * shell_radius * self.parameters['shell']['density'])

                surface = self.surface_distribution(generate, shell, atom)

                #scale and translate to atom center
                surface *= shell_radius
                surface += self.scf.mol.atom[atom].center[:] * CONSTANTS('bohr->angstrom')

                valid = []
                #remove points inside other atoms
                for i in range(surface.shape[0]):
                    valid.append(not False in [external(i,j) for j in range(self.scf.mol.natm) if atom != j])
                surface = surface[valid,:]

                global_surface = np.vstack((global_surface, surface)) if global_surface is not None else surface

                #mark end of each atom data in global surface distribution
                self.atom_index[atom] = global_surface.shape[0]

        self.surface = global_surface

    def view_surface(self, azimuth=60, elevation=60):
        #generate a plot of the surface

        fig = p.figure()
        ax = fig.add_subplot(111, projection='3d')

        color = 'w'
        color_dict = {'H':'gray','O':'r','C':'k','N':'b','Cl':'g','F':'m','S':'y'}

        for i in range(self.scf.mol.natm):
            block = slice(0 if i == 0 else int(self.atom_index[i-1]),  int(self.atom_index[i]-1))

            if self.scf.mol.atom[i].symbol in color_dict: color = color_dict[self.scf.mol.atom[i].symbol]
            ax.scatter3D(self.surface[block,0], self.surface[block,1], self.surface[block,2], color=color)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo['grid']['color'] = (0, 0, 0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0

        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(azim=azimuth, elev=elevation)
        p.show()

    def get_inverse_radii(self):
        #make array of inverse radii of each point to each atom - Bohr

        inverse_radii = np.zeros((self.surface.shape[0], self.scf.mol.natm))
        for p in range(self.surface.shape[0]):
            for atom in range(self.scf.mol.natm):
                inverse_radii[p, atom] = np.linalg.norm(self.surface[p][:]/CONSTANTS('bohr->angstrom')
                                                      - self.scf.mol.atom[atom].center[:] )
        return np.reciprocal(inverse_radii)

    def generate_potential_field(self):
        #generate the electrostatic field on the shells

        self.electrostatic_potential = np.zeros(self.surface.shape[0])
        for p in range(self.surface.shape[0]):
            components = self.electric_field(self.surface[p, :] / CONSTANTS('bohr->angstrom'))

            #build potential [3] is electric component and [7] is nuclear component
            self.electrostatic_potential[p] = components[7] - components[3]

    def electric_field(self, point):
        #compute expectation values for electric field components and electric potential
        #[0-2] - electric field strength, [3] - electric field potential,
        #[4-6] - nuclear field strength [7] - nuclear field potential

        electric_field_components = np.zeros(8)

        #electric components
        electric_field_components[:4] = 2.0 * np.einsum('pr,xrp', self.scf.get('d'),
                                        aello(self.scf.mol.atom, self.scf.mol.orbital, mode='electric field', gauge=point),
                                        optimize=True)
        #nuclear components
        gauge_coordinates = np.array([x.center - point for x in self.scf.mol.atom])
        radial_distance = [np.linalg.norm(gauge_coordinates[i]) for i in range(self.scf.mol.natm)]

        for i, atom in enumerate(self.scf.mol.atom):

            if radial_distance[i] < 1e-10: continue

            electric_field_components[4:7] = [atom.number * gauge_coordinates[i,x]/pow(radial_distance[i], 3) for x in range(3)]
            electric_field_components[7] += atom.number / radial_distance[i]

        return electric_field_components

    def build_constraints(self):
        #if we have constraints on the ESP implement them

        constraint_charge, constraint_indices  = [], []
        self.constraints_count, self.constraint_type = 0, []

        for constraint in self.parameters['constrain']:

            if constraint[0] == '=':
                for i in range(len(constraint[1]) - 1):
                    constraint_charge.append(0)
                    self.constraint_type.append('=')
                    constraint_indices.append([constraint[1][i], constraint[1][i+1]])
                    self.constraints_count += 1
            else:
                constraint_charge.append(constraint[0])
                self.constraint_type.append('+')
                constraint_indices.append(constraint[1])
                self.constraints_count += 1

        self.constraint_charge, self.constraint_indices = constraint_charge, constraint_indices

    def build_matrix(self, inverse_radii):
        #matrix is a natm dimension block in top left, a row and column of length natm of 1's
        #and number of constrains rows and columns to the right and bottom

        natm= self.scf.mol.natm
        ndim = natm + self.constraints_count + 1

        a = np.zeros((ndim ,ndim))
        a[:natm, :natm] = np.einsum('ki,kj->ij', inverse_radii, inverse_radii, optimize=True)
        a[:natm, natm], a[natm, :natm] = 1.0, 1.0

        return a

    def build_vector(self, inverse_radii):
        #build vector for solve equation

        natm = self.scf.mol.natm
        ndim = natm + 1 + self.constraints_count

        b = np.zeros(ndim)
        b[:natm] = np.einsum('p,pi->i', self.electrostatic_potential, inverse_radii, optimize=True)
        b[natm] = self.scf.mol.charge

        return b

    def generate_constrained_charge(self, matrix, vector):
        #evaluate charges with any constrained charges applied

        natm = self.scf.mol.natm

        #add applied constraints
        for constraint in range(self.constraints_count):

            vector[natm + 1 + constraint] = self.constraint_charge[constraint]

            if self.constraint_type[constraint] == '+':
                sgn = [1] *  len(self.constraint_indices[constraint])
            else:
                sgn = [-1] + [1] * (len(self.constraint_indices[constraint]) - 1)

            for n, i in enumerate(self.constraint_indices[constraint]):
                matrix[natm + 1 + constraint, i - 1] = sgn[n]
                matrix[i - 1, natm + 1 + constraint] = sgn[n]

        #remove zero rows or columns from a
        matrix = matrix[~np.all(matrix == 0, axis=1)]
        matrix = matrix[~np.all(matrix == 0, axis=0)]

        return matrix, vector

    def solve_constrained_charges(self, matrix, vector):
        #solve the system of equations for constrained charges
        try:
            q = np.linalg.solve(matrix, vector)
        except:
            exit('Linalg Solve failed')

        self.constrained_charges = q[:self.scf.mol.natm]

    def solve_restrained_charges(self, matrix, vector, q):
        #solve the system for restrained charges

        cycle_q = q[:]

        natm = self.scf.mol.natm

        for cycle in range(self.parameters['restrain']['cycles']):

            a = self.hyperbolic_restraint(matrix, q)
            q = np.linalg.solve(a, vector)

            #test convergence
            if np.sqrt( max(pow(q[:natm] - cycle_q[:natm],2))) < self.parameters['restrain']['tol']:
                self.restrained_charges = q[:natm]
                break
            cycle_q = q.copy()

        else:
            self.restrained_charges = None

        return cycle

    def hyperbolic_restraint(self, matrix, q):
        #apply Hyperbolic restrain

        a = matrix.copy()

        for i, x in enumerate(self.scf.mol.atom):

            if not self.parameters['restrain']['H'] or x.symbol != 'H' :
                a[i, i] = matrix[i, i] + (self.parameters['restrain']['a']
                                       / np.sqrt(q[i] * q[i]
                                       + self.parameters['restrain']['b'] * self.parameters['restrain']['b']))

        return a

    def get_carbon_groups(self):
        #find all carbons with attached Hydrogen atoms

        self.carbon_groups = []
        for atom in self.scf.mol.atom:

            if atom.symbol == 'C':
                attached_hydrogen = []

                #get C-H connections
                self.carbon_groups.append([atom.id + 1, [i + 1 for i, x in enumerate(self.scf.mol.atom)
                                                        if x.symbol == 'H' and is_bond(self.scf, x, atom)]])


    def refine_charges(self):
        #if stage 2 refine is requested apply

        #process refine parameters
        atoms = list(range(1, self.scf.mol.natm + 1))

        self.parameters['constrain'] = []
        #for each CH grouping constrain hydrogens to be equal ['=', [n,m,..]]
        for group in self.carbon_groups:
            self.parameters['constrain'].append(['=', group[1]])

            for i in group[1]:
                atoms.remove(i)
            atoms.remove(group[0])

        #other atoms take restained charge
        for i in atoms:
            self.parameters['constrain'].append([float(self.restrained_charges[i-1]),[i]])

    def classical_charges(self, q, inverse_radii):
        #root-mean-square deviation from classical result

        delta = lambda x: np.einsum('x,x->', q, inverse_radii[x, :]) - self.electrostatic_potential[x]

        return np.sqrt(sum([delta(x)**2.0 for x in range(inverse_radii.shape[0])])/inverse_radii.shape[0])

    def assert_charges(self):
        #checks on charge sums

        for x in self.cache.keys():
            if ('constrained' in x) or ('restrained' in x):
                assert np.isclose(sum(self.cache[x]), self.scf.mol.charge)

