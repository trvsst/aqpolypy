"""
:module: Bjerrum
:platform: Unix, Windows, OS
:synopsis:

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import expi
import aqpolypy.units.units as un


class Bjerrum:

    def __init__(self, water_object):
        """
        constructor

        :param water_object: object of water class
        :instantiate: water class object, bjerrum length

        """
        self.epsilon = water_object.dielectric_constant()
        self.e_square = un.e_square()
        self.kbt = water_object.tk * un.k_boltzmann()

        # Bjerrum length in A
        self.bjerrum_length = un.m_2_angstrom(self.e_square / (self.kbt * self.epsilon))

        # molar volume
        self.molar_volume = water_object.molar_volume()

    def temp_star(self, ion_size):
        """
            Dimensionless temperature according to Ebeling :cite:`Ebel:71`

            .. math::
                :label: dimensionless_temp

                T^{\\ast}=\\frac{a}{l_B(T)}

            Bjerrum length is

            .. math::

                l_B(T)=\\frac{q^2}{4\\pi\\varepsilon_0\\varepsilon_r k_B T}

            :param ion_size: Ion size (in Angstrom)
            :return: dimensionless temperature (float)
            """

        return ion_size / self.bjerrum_length

    def bjerrum_constant_approx(self, ion_size, acc=1e-2):
        """
        Value of Bjerrum constant calculated by integral according to Ebeling :cite:`Ebel:71`

        .. math::
            :label: association_constant

            K_B(T)=4\\pi a^3\\int_1^{\\infty}dy y^2\\left(e^{b/y}+e^{-b/y}-2-\\frac{b^2}{y^2}\\right)

        B variable is

        .. math::

            b = 1/T^{\\ast}

        Ion radius is

        .. math::

            a

        :param ion_size: Ion size (in Angstrom)
        :param acc: Accuracy
        :return: Bjerrum constant in units of :math:`a^3` (float)
        """

        b_par = 1 / self.temp_star(ion_size)
        max_val = 1 / acc

        def intgrnd(x):
            z = b_par / x
            return x ** 2 * (np.exp(z) + np.exp(-z) - 2 - z ** 2)

        integral_quad = quad(intgrnd, 1, max_val)
        val = integral_quad[0]
        err = integral_quad[1]

        err_cut_off = (acc * b_par ** 4 / 12.0) / val

        bjerrum_constant_approx = (4 * np.pi * val, err_cut_off * val, err_cut_off, err)
        return bjerrum_constant_approx

    def bjerrum_constant(self, ion_size):
        """
        Value of Bjerrum constant according to Ebeling :cite:`Ebel:71`

        :param ion_size: Ion size (in Angstrom)
        :return: Bjerrum constant in units of :math:`a^3` (float)
        """

        z = 1 / self.temp_star(ion_size)

        t1 = z ** 4 * np.exp(-z) * (expi(z) - expi(-z) + 6 / z + 4 / z ** 3)
        t2 = -z * (z ** 2 + z + 2)
        t3 = -z * (z ** 2 - z + 2) * np.exp(-2 * z)

        bjerrum_constant = 4 * np.pi * (t1 + t2 + t3) * np.exp(z) / (6 * z)
        return bjerrum_constant

    def bjerrum_constant_according_bjerrum(self, ion_size):
        """
        Value of Bjerrum constant according to Bjerrum

        :param ion_size: Ion size (in Angstrom)
        :return: Bjerrum constant in units of :math:`a^3` (float)
        """

        z = 1 / self.temp_star(ion_size)
        def intgrnd(x):
            return np.exp(x)/x**4

        max_val = z

        if max_val < 2:
            return ion_size

        integral_quad = quad(intgrnd, 2, max_val)
        val = integral_quad[0]

        return 4*np.pi*z**3*val

    def bjerrum_constant_mol_litre(self, ion_size, q_valence, bjerrum_original=False):
        """
        Value of Bjerrum constant according to Bjerrum

        :param ion_size: Ion size (in Angstrom)
        :param q_valence: tuple with the valence of both ions
        :param bjerrum_original: if True return original Bjerrum value, if false the result by Ebeling
        :return: Bjerrum constant in units of mols/litre (float)
        """

        q_p, q_m = q_valence

        m_size = ion_size/np.abs(q_p*q_m)

        if bjerrum_original:
            kb = self.bjerrum_constant_according_bjerrum(m_size)
        else:
            kb = self.bjerrum_constant(m_size)

        conv = 1e-3*un.mol_angstrom_2_mol_mcube(1)
        
        return kb*(ion_size)**3/conv

    def b_parameter(self):
        """
        This is the quantity :math:`\\kappa_0` inverse of the Debye-length at ionic strength 1

        This quantity defines the size of the ion in the formula for the activity

        """

        v_r = 1 / (1e3 * self.molar_volume)
        v_w = 1 / un.mol_lit_2_mol_angstrom(v_r)

        return np.sqrt(8*np.pi*self.bjerrum_length/(un.delta_w()*v_w))
