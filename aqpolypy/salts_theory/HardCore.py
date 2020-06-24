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

    def __init__(self):
        """
        constructor
        """

    @staticmethod
    def free_energy_hc_excess(c, ion_size, b_fac=np.array([1, 1, 2])):
        """
        Excess free energy of the hard core according to :cite:`Levin1996`

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :param b_fac: B_factors for the different species in units of ion_size
        :return : excess free energy (float)
        """

        cc_molecular = un.mol_lit_2_mol_angstrom(c)

        arg_vol = np.dot(b_fac, cc_molecular) * ion_size ** 3

        free_energy_hc_excess = np.sum(cc_molecular, axis=0) * np.log(1 - arg_vol)
        return free_energy_hc_excess

    @staticmethod
    def pot_chem_hc_excess(c, ion_size, b_fac=np.array([1, 1, 2])):
        """
        Excess chemical potential of the hard core according to :cite:`Levin1996`

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :param b_fac: B_factors for the different species in units of ion_size
        :return : excess chemical potential (float)
        """

        cc_molecular = un.mol_lit_2_mol_angstrom(c)

        arg_vol = np.dot(b_fac, cc_molecular) * ion_size ** 3

        t1 = -np.log(1 - arg_vol)
        t2 = ion_size ** 3 * np.sum(cc_molecular, axis=0) / (1 - arg_vol)

        pot_chem_hc_excess = np.outer(b_fac, t2) + t1
        return pot_chem_hc_excess

    @staticmethod
    def pressure_hc_excess(c, ion_size, b_fac=np.array([1, 1, 2])):
        """
        Excess pressure for the hard core according to :cite:`Levin1996`

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :param b_fac: B_factors for the different species in units of ion_size
        :return : excess pressure (float)
        """

        cc_molecular = un.mol_lit_2_mol_angstrom(c)

        arg_vol = np.dot(b_fac, cc_molecular) * ion_size ** 3

        pressure_hc_excess = np.sum(cc_molecular, axis=0) / (1 - arg_vol)
        return pressure_hc_excess
