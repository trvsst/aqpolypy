"""
:module: WaterMilleroBP
:platform: Unix, Windows, OS
:synopsis: Derived water properties class utilizing Fine & Millero and Bradley & Pitzer calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Changed dielectric calculations from Archer & Wang to Bradley & Pitzer
..                  - Added member method a_phi and a_v
..                  - Implemented Bradley & Pitzer calculations for a_phi and a_v in constructor
"""

import numpy as np
import math
import aqpolypy.units.units as un
import aqpolypy.water.WaterPropertiesABC as wp


class WaterPropertiesFineMillero(wp.WaterProperties):
    """
    Water Properties following the work of Fine and Millero :cite:`Fine1973`
    Dielectric constant from Bradley and Pitzer :cite:`Bradley1979`
    Apparent molal volume from Bradley and Pitzer :cite:`Bradley1979`
    Osmotic coefficient from Bradley and Pitzer :cite:`Bradley1979`

    """

    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, molecular weight, polarizability, dipole moment

        """
        super().__init__(tk, pa)

        """
        Calculations for density
        """
        self.t = self.tk - un.celsius_2_kelvin(0)
        self.y = un.atm_2_bar((self.pa - 1))

        self.den1 = 0.9998396 + self.t * 18.224944e-3 - 7.922210e-6 * self.t ** 2 - 55.44846e-9 * self.t ** 3
        self.den2 = 149.7562e-12 * self.t ** 4 - 393.2952e-15 * self.t ** 5

        self.V0 = (1 + self.t * 18.159725e-3) / (self.den1 + self.den2)

        self.B = 19654.320 + 147.037 * self.t - 2.21554 * self.t ** 2 + 1.0478e-2 * self.t ** 3 - 2.2789e-5 * self.t ** 4
        self.a1 = 3.2891 - 2.3910e-3 * self.t + 2.8446e-4 * self.t ** 2 - 2.82e-6 * self.t ** 3 + 8.477e-9 * self.t ** 4
        self.a2 = 6.245e-5 - 3.913e-6 * self.t - 3.499e-8 * self.t ** 2 + 7.942e-10 * self.t ** 3 - 3.299e-12 * self.t ** 4

        # this in cm^3/gram
        self.vol = self.V0 * (1 - (self.y / (self.B + self.a1 * self.y + self.a2 * self.y ** 2)))

        # density in kg/m^3
        self.rt = 1e3 / self.vol

        # Calculations for molar volume

        self.molVol = 1e-3 * self.MolecularWeight / self.rt

        # Calculations for dielectric constant

        # coefficients
        self.U = np.zeros(9)
        self.U[0] = 3.4279e2
        self.U[1] = -5.0866e-3
        self.U[2] = 9.4690e-7
        self.U[3] = -2.0525
        self.U[4] = 3.1159e3
        self.U[5] = -1.8289e2
        self.U[6] = -8.0325e3
        self.U[7] = 4.2142e6
        self.U[8] = 2.1417

        self.D100 = self.U[0] * (math.e ** (self.U[1] * self.tk + self.U[2] * self.tk ** 2))
        self.C = self.U[3] + self.U[4] / (self.U[5] + self.tk)
        self.B_dc = self.U[6] + (self.U[7] / self.tk) + self.U[8] * self.tk

        self.dielectricConstant = self.D100 + self.C * np.log((self.B_dc + un.atm_2_bar(self.pa)) / (self.B_dc + 1000))
        self.dielectricConstant_der = un.atm_2_bar(1) * self.C / (self.B_dc + (un.atm_2_bar(self.pa)))

        #  Calculations for compressibility

        self.beta = self.V0 * (self.B - self.a2 * self.y ** 2) / (
                self.vol * (self.B + self.a1 * self.y + self.a2 * self.y ** 2) ** 2)
        self.comp = un.atm_2_bar(self.beta)

        # Calculations for osmotic coefficient

        # Bjerrum length
        self.l_b = un.e_square() / (un.k_boltzmann() * self.tk * self.dielectricConstant)

        self.fac = 2 * np.pi * un.avogadro() * self.rt * self.l_b ** 3

        self.osmotic_coefficient = np.sqrt(self.fac) / 3

        # Calculations for apparent molal volume

        # add a factor 1e6 and convert to Pascal (lat two parts of a_v_0)
        self.a_v_0 = 2 * self.osmotic_coefficient * un.r_gas() * self.tk * 1e6 / un.atm_2_pascal(1)
        self.a_v_1 = self.dielectricConstant
        self.a_v_2 = 3 * self.dielectricConstant_der
        self.a_v_3 = self.comp

        self.app_molal_vol = self.a_v_0 * (self.a_v_2 / self.a_v_1 - self.a_v_3)

    def density(self):
        """
            Water density according to Fine Millero :cite:`Fine1973`

            restricted to temperatures in range [0, 100]

            :return: water density in SI
            :rtype: float


            TODO: include warning if the temperature is outside [0,100]
            """
        # if (self.tk > 100 or self.tk < 0):
        #    return print("Temperature out of allowable range: [0, 100]")

        return self.rt

    def molar_volume(self):
        """
            Molar volume of water according to Fine and Millero :cite:`Fine1973`

            restricted to temperatures in range [0, 100]

            :return: water molar volume in SI
            :rtype: float

            TODO: include warning if the temperature is outside [0,100]
            """
        # if (T > 100 or T < 0):
        #    return print("Temperature out of allowable range: [0, 100]")

        return self.molVol

    def dielectric_constant(self):
        """
            Dielectric constant according to Bradley and Pitzer :cite:`Bradley1979`

            :return: dielectric constant in SI
            :rtype: float
            """
        return self.dielectricConstant

    def compressibility(self):
        """
            Water compressibility according to Fine and Millero :cite:`Fine1973`

            :return: compressibility of water in SI
            :rtype: float
            """
        return self.comp

    def a_phi(self):
        """
            Osmotic coefficient according to Bradley and Pitzer :cite:`Bradley1979`

            :return: osmotic coefficient in SI
            :rtype: float
            """
        return self.osmotic_coefficient

    def a_v(self):
        """
            Apparent molal volume according to Bradley and Pitzer :cite:`Bradley1979`

            :return: apparent molal volume in SI
            :rtype: float
            """
        return self.app_molal_vol
