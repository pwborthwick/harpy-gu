![image](https://user-images.githubusercontent.com/73105740/224044727-10d94d6e-ae70-45ae-8c58-f58187aabe0a.png) <h1>Harpy-gu</h1>



<ins>**Introduction**</ins>
    Harpy (standing for [**Har**tree-Fock **Py**thon)](https://github.com/pwborthwick/harpy) is a collection of quantum chemistry codes. The original aim was to code as close to pseudo-code as was possible so that the algorithms could be understood by any programmer. As the project evolved this aim lapsed somewhat but most of the core modules are written that way. This program is a **g**rown-**u**p version of *harpy* using a more 'pythonic' programming style.

<ins>**Getting Started**</ins>
    The easiest way to install the program is simply copy the **harpy-gu** folder and its sub-folders to your local machine. From the *harpy-gu* root directory run ```python setup.py build_ext --inplace install --user``` which will cythonize the integral module. Then the program is ready to run by creating a file with a python extension (see for example 'examples/01-rhf.py') and run from the root directory. I use a ```export PYTHONPATH=<harpy-gu root directory>```. There is no documentation but many annotated examples are given in the ```examples``` folder.

<ins>**Directory Structure**</ins>

    \scf      - this contains modules for restricted Hartree-Fock (rhf), unrestricted Hartree-Fock (uhf), restricted open
                Hartree-Fock (rohf), restricted Kohn-Sham (rks) and unrestricted Kohn-Sham (uks).

    \mol      - this contains the modules for molecule and basis definition (mol) and various utilities (utl)

    \int      - this contains the modules for Gaussian integrals (aello), Lebedev grid and numerical quadrature (grd),
                DFT functional definitions (ksn) and routines for transformations between atomic, molecular and spin
                basis (mo_spin).

    \bas      - this contains a few basis sets. The files are in psi4 format from BSE.

    \examples - these are the examples of how to use the software

    \phf      - this contains the post Hartree-Fock methods. Included are algebraic diagrammatic construction (adc),
                coupled-cluster theory (cct), configuration interaction (cit), eigenvalue solvers (eig), equation of motion
                coupled-cluster (eom), electron propogator theory (ept), full configuration interaction (fci), G0W0@HF (gwt),
                Moller-Plesset theory (mpt), restrained electrostatic potential (rESP) and time-dependent Hartree-Fock (tdhf)

<ins>**Selected Contents**</ins>

**mol.mol** - constants definition, z-file creation, reading basis files.

**mol.utl**   - Pade approximates, spectrum analysis, integrators for BOMD, bond determination (is_bond), Thompson spherical distribution. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spectra broadening wave shapes.

**int.aello** - Cython integrals: overlap, 1e kinetic and coulomb, 2e repulsion, dipole, nabla, angular momentum and electric field. Force (gradient) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;integrals for overlap, 1e kinetic and coulomb, 2e repulsion.

**int.grd** - Lebedev grids, Mura-Knowles, Treutler pruning, Stratmann-Becke scheme, Aldrich-Treutler adjustment using BRAGG radii, atomic grid &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;construction.

**int.ksn** - evaluation of orbitals and density on a grid, evaluation of fuctional on grid, LDA exchange, VWN-RPA correlation for spin unpolarized
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and spin-polarized method.

**int.mo_spin** - transforms to molecular and molecular spin basis from atomic basis, orbital energy difference tensors.

**scf.diis** - direct inversion of iterative subspace for scf in closed and open methods and for coupled-cluster.

**scf.out** - output formatting fror all scf methods.

**scf.rhf** - restricted Hartree-Fock code. SCF calculation and analysis of dipoles, Mulliken charges and Mayer bond orders.

**scf.uhf** - restricted Hartree-Fock code. SCF calculation and analysis of dipoles, Mulliken charges and Mayer bond orders. Maximum Overlap &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Method (MOM).

**scf.rohf** - restricted-open Hartree-Fock. SCF calculation and analysis of dipoles, Mulliken charges and Mayer bond orders.

**scf.rks** - restricted Kohn-Sham code.  SCF calculation and analysis of dipoles, Mulliken charges and Mayer bond orders. LDA excahage and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VWN_RPA
              correlation.

**scf.uks** - unrestricted Kohn-Sham code.  SCF calculation and analysis of dipoles, Mulliken charges and Mayer bond orders. LDA excahage and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VWN_RPA
              correlation.

**phf.adc** - algebraic diagrammatic construction (2) EE/IP/EA, transition properties, state analysis.

**phf.cct** - spin coupled-cluster methods. CCD, CCSD, CCSD(T), LCCD, LCCSD, CCSDT-1a, CCSDT-1b and QCISD. &Lambda;-CCSD and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;reduced density matrices. Restricted spin CCD and CCSD.

**phf-cit** - configuration interaction. spin and spin-adapted de,terminant based FCI from phf.fci module, CIS, spin-adapted CIS , Random Phase &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Approximation (block, linear, Hermitan and Tamm-Dancoff Approximation), CIS-MP2, CIS(D), transition properties.

**phf-eig** - eigensolvers. Direct solvers via numpy and scipy. Davidon iterative solver for many root and a single targetted root version.

**phf.eom** - equation-of-motion CCSD EE/IP/EA. L and R eigenvectors. State analysis.

**phf.ept** - electron propogator theory. EP2. EP3 and Approximate Greens Function.

**phf.fci** - determinant basd full configuration interaction. Spin and spin-restricted modes. CIS, CISD, CISDT, CISDTQ and CISDTQP explicitly. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Residues.

**phf.gwt** - Many-Body Perturbation Theory GW approximation.Some simple  G0W0@HF.

**phf.mpt** - Moller-Plesset perturbation theory. Spin MP2, MP3 and spin restricted MP2 (spin-component scaled). MP2 relaxed and unrelaxed &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;density matrices and dipoles. Orbital Optimized MP2, Laplace transform spin-restricted MP2, natural orbitals.

**phf.rESP** - restrained Electrostatic Potentials. point distribution on spherical surface by Connolly, Fibonacci, Saff & Kuijlaars and Thomson. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Constrained, restrained and carbon group refinement.

**phf.tdhf** - time-dependent Hartree-Fock. Transition properties - electric and magnetic dipoles in length and velcity gauges. Oscillator &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and rotatary strengths. Optical properties OPA and ECD. Real-time TDHF using Magnus (2) propogator.

<ins>**Example Scripts**</ins>

There are a number (21) of examples of using the modules in *harpy-gu*, the files are heavily annotated and togther with the source should give a good idea of how to get results from the code. Where possible the code has been checked against either PySCF or Psi4 for accuracy. I hope there is something useful here for people building their own code (even if it's how not to code it!)

