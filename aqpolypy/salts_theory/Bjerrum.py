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
import aqpolypy.water.WaterMilleroBP as wfm
import aqpolypy.units.units as un

class Bjerrum():



    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, bjerrum length

        """
        # temperature and pressure
        self.tk = tk
        self.pa = pa

        self.epsilon = wfm.WaterPropertiesFineMillero(self.tk).dielectric_constant()
        self.e_square = un.e_square()
        self.kbt = self.tk * un.k_boltzmann()

        # Bjerrum length in A
        self.bjerrum_length = un.m_2_angstrom(self.e_square / (self.kbt * self.epsilon))

    def temp_star(self, ion_size):
        """
            Dimensionless temperature according to Ebeling :cite:`Ebel:71`

            .. math::
                :label: dimensionless_temp

                T^{\\ast}=\\frac{a}{l_B(T)}

                :label: bjerrum_length

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

            :label: b_variable

            b=1/T^{\\ast}

            :label: ion_radius

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

        val, err = quad(intgrnd, 1, max_val)

        err_cut_off = (acc * b_par ** 4 / 12.0) / val

        bjerrum_constant_approx = (4 * np.pi * val, err_cut_off * val, err_cut_off, err)
        return bjerrum_constant_approx

    def bjerrum_constant(self, ion_size):
        """
        Value of Bjerrum constant according to Ebeling :cite:`Ebel:71`

        :param ion_size: Ion size (in Angstrom)
        :return: Bjerrum constant in units of :math:`a^3`(float)
        """

        z = 1 / self.temp_star(ion_size)

        t1 = z ** 4 * np.exp(-z) * (expi(z) - expi(-z) + 6 / z + 4 / z ** 3)
        t2 = -z * (z ** 2 + z + 2)
        t3 = -z * (z ** 2 - z + 2) * np.exp(-2 * z)

        bjerrum_constant = 4 * np.pi * (t1 + t2 + t3) * np.exp(z) / (6 * z)
        return bjerrum_constant
