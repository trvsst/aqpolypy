"""
:module: DebyeHuckel
:platform: Unix, Windows, OS
:synopsis:

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
import aqpolypy.units.units as un


class DebyeHuckel:

    def __init__(self, bjerrum_object):
        """
        constructor

        :param bjerrum_object: object of water class
        :instantiate: Bjerrum class object
        """

        self.bjerrum_object = bjerrum_object

    def debye_length(self, c):
        """
            Debye length according to :cite:`Levin1996`

            .. math::
                :label: debye_length

                \\lambda_D = \\frac{1}{\\sqrt{4\\pi l_B(T)n_1}}

            :param c: Concentration of ions (both + and -) mols/litre (M)
            """
        c_m = un.mol_lit_2_mol_angstrom(c)

        debye_length = 1.0 / np.sqrt(4 * np.pi * c_m * self.bjerrum_object.bjerrum_length)
        return debye_length

    def ion_size_debye_ratio(self, c, ion_size):
        """
        Helper function computes ion size divided by debye length

        :param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: Ion diameter (in Angstrom)
        :return: excess free energy per unit volume in units of math:'k_BT/a^3' (float)
        """

        return ion_size / self.debye_length(c)

    def free_energy_db_excess(self, c, ion_size):
        """
        Debye Huckel Free energy according to :cite:`Levin1996`

        :param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: Ion diameter (in Angstrom)
        :return: excess free energy per unit volume in units of math:'k_BT/a^3' (float)
        """

        z = self.ion_size_debye_ratio(c, ion_size)

        free_energy_db_excess = -(np.log(1 + z) - z + 0.5 * z ** 2) / np.pi
        return free_energy_db_excess

    def pot_chem_db_excess(self, c, ion_size):
        """
        Excess chemical potential according to :cite:`Levin1996`

        :param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: Ion diameter (in Angstrom)
        :return: excess chemical potential in units of math:'k_BT' (float)
        """
        z = self.ion_size_debye_ratio(c, ion_size)

        temp_star = self.bjerrum_object.temp_star(ion_size)

        pot_chem_db_excess = -0.5 * z / (temp_star * (1 + z))
        return pot_chem_db_excess

    def pressure_db_excess(self, c, ion_size):
        """
        Excess pressure according to :cite:`Levin1996`

        :param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: Ion diameter (in Angstrom)
        :return: excess pressure in units of math:'k_BT/a^3' (float)
        """
        z = self.ion_size_debye_ratio(c, ion_size)

        pressure_db_excess = (np.log(1 + z) - 0.5 * z ** 2 / (1 + z)) / (4 * np.pi)
        return pressure_db_excess
