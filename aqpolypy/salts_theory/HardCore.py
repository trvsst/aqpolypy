"""
:module: HardCore
:platform: Unix, Windows, OS
:synopsis:

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
import aqpolypy.units.units as un


class HardCore:

    def __init__(self, b_fac=np.array([1, 1, 2])):
        """
        constructor

        :param b_fac: B_factors for the different species in units of ion_size
        :instantiate: B_factors
        """
        self.b_fac = b_fac

    def free_energy_hc_excess(self, c, ion_size):
        """
        Excess free energy of the hard core

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :return: excess free energy (float)
        """

        cc_molecular = un.mol_lit_2_mol_angstrom(c)

        arg_vol = np.dot(self.b_fac, cc_molecular) * ion_size ** 3

        free_energy_hc_excess = np.sum(cc_molecular, axis=0) * np.log(1 - arg_vol)
        return free_energy_hc_excess

    def pot_chem_hc_excess(self, c, ion_size):
        """
        Excess chemical potential of the hard core

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :return: excess chemical potential (float)
        """

        cc_molecular = un.mol_lit_2_mol_angstrom(c)

        arg_vol = np.dot(self.b_fac, cc_molecular) * ion_size ** 3

        t1 = -np.log(1 - arg_vol)
        t2 = ion_size ** 3 * np.sum(cc_molecular, axis=0) / (1 - arg_vol)

        pot_chem_hc_excess = np.outer(self.b_fac, t2) + t1
        return pot_chem_hc_excess

    def pressure_hc_excess(self, c, ion_size):
        """
        Excess pressure for the hard core

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :return: excess pressure (float)
        """

        cc_molecular = un.mol_lit_2_mol_angstrom(c)

        arg_vol = np.dot(self.b_fac, cc_molecular) * ion_size ** 3

        pressure_hc_excess = np.sum(cc_molecular, axis=0) / (1 - arg_vol)
        return pressure_hc_excess
