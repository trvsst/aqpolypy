"""
:module: Dipole
:platform: Unix, Windows, OS
:synopsis:

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.salts_theory.DebyeHuckel as dh


class Dipole:

    def __init__(self, bjerrum_object):
        """
        constructor

        :param bjerrum_object: object of water class
        :instantiate: Bjerrum class object
        """

        self.bjerrum_object = bjerrum_object

    @staticmethod
    def omega(x):
        """
        Function :math:`\\omega` as defined in :cite:`Levin1996` Eq. 7.13

        :param x: argument
        :return: value of :math:`\\omega` (float)
        """
        omega = 3 * (np.log(1 + x + x ** 2 / 3) - x + x ** 2 / 6) / x ** 4
        return omega

    @staticmethod
    def der_x2omega(x):
        """
        Derivative of :math:`x^2\\omega`

        :param x: argument
        :return: value of the derivative (float)
        """

        val1 = x * (6 + 5 * x + x ** 2) / (3 + 3 * x + x ** 2)
        val2 = -2 * np.log(1 + x + x ** 2 / 3)

        der_x2omega = 3 * (val1 + val2) / x ** 3
        return der_x2omega

    def free_energy_dp_excess(self, c, ion_size, x_pair, a1_d=1, a2_d=1):
        """
        Excess free energy of the dipole function

        :param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :param x_pair: fraction of Bjerrum pairs
        :param a1_d: dipole distance, in units of ion_size
        :param a2_d: diameter of enclosing dipole, in units of ion_size
        :return: excess free energy in units of :math:`ion\\_size^3` (float)
        """

        l_dh = dh.DebyeHuckel(self.bjerrum_object).debye_length((1 - x_pair) * c)
        rho2 = un.mol_lit_2_mol_angstrom(c) * x_pair * 0.5 * ion_size ** 3
        z = ion_size / l_dh
        t_star = self.bjerrum_object.temp_star(ion_size)

        f_dip = -z ** 2 * a1_d ** 2 * rho2 * self.omega(z * a2_d) / (t_star * a2_d)

        return f_dip

    def pot_chem_dp_excess(self, c, ion_size, x_pair, a1_d=1, a2_d=1):
        """
        Excess chemical potential of the dipole function

        :param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :param x_pair: fraction of Bjerrum pairs
        :param a1_d: dipole distance, in units of ion_size
        :param a2_d: diameter of enclosing dipole, in units of ion_size
        :return: excess chemical potential (ndarray)
        """

        l_dh = dh.DebyeHuckel(self.bjerrum_object).debye_length((1 - x_pair) * c)
        rho = un.mol_lit_2_mol_angstrom(c) * ion_size ** 3
        rho1 = 0.5 * rho * (1 - x_pair)
        rho2 = 0.5 * rho * x_pair
        z = ion_size / l_dh
        t_star = self.bjerrum_object.temp_star(ion_size)

        mu_0 = -z ** 2 * a1_d ** 2 * self.omega(z * a2_d) / (t_star * a2_d)

        val1 = -z ** 2 * a1_d ** 2 * rho2 * self.omega(z * a2_d) / (t_star * a2_d ** 3)
        val2 = self.der_x2omega(z * a2_d) * z * a2_d

        mu_p = val1 * val2 * 0.5 * rho1

        return np.array([mu_p, mu_p, mu_0])
