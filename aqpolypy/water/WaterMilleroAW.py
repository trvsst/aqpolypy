"""
:module: WaterMilleroAW
:platform: Unix, Windows, OS
:synopsis: Derived water properties class utilizing Fine & Millero and Archer & Wang calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added constructor with temperature and pressure parameters.
..                  - Updated member methods to use attributes and functions from units.
..                  - Moved calculations from member methods into the constructor.
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.water.WaterPropertiesABC as wp


class WaterPropertiesFineMillero(wp.WaterProperties):
    """
    Water properties following the work of Fine and Millero :cite:`Fine1973`
    Dilectric constant from Archer and Wang :cite:`Archer1990`

    """

    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, molecular weight, polarizability, dipole moment

        """
        super().__init__(tk, pa)

        # Calculations for density

        self.t = self.tk - un.celsius_2_kelvin(0)
        # pressure in applied bar according to Fine and Millero
        self.y = un.atm_2_bar(self.pa - 1)

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

        # convert from atmospheres to MPa
        self.p_mpa = 1e-6 * un.atm_2_pascal(self.pa)

        # coefficients
        self.b = np.zeros(9)
        self.b[0] = -4.044525e-2
        self.b[1] = 103.6180
        self.b[2] = 75.32165
        self.b[3] = -23.23778
        self.b[4] = -3.548184
        self.b[5] = -1246.311
        self.b[6] = 263307.7
        self.b[7] = -6.928953e-1
        self.b[8] = -204.4473

        def g(x, y):
            t1 = self.b[0] * y / x + self.b[1] / np.sqrt(x) + self.b[2] / (x - 215) + self.b[3] / np.sqrt(x - 215)
            t2 = self.b[4] / (self.tk - 215) ** 0.25
            t3 = np.exp(self.b[5] / x + self.b[6] / x ** 2 + self.b[7] * y / x + self.b[8] * y / x ** 2)
            return 1 + 1e-3 * self.rt * (t1 + t2 + t3)

        self.fac = un.one_over4pi_epsilon0() * 4 * np.pi * self.mu ** 2 / (3 * un.k_boltzmann() * self.tk)
        self.b_fac_0 = self.fac * g(self.tk, self.p_mpa)
        self.b_fac_1 = self.alpha + self.b_fac_0
        self.b_fac = self.rt * un.avogadro() * self.b_fac_1 / (3.0 * self.MolecularWeight / 1000)

        self.dielectricConstant = 0.25 * (1 + 9 * self.b_fac + 3 * np.sqrt(1 + 2 * self.b_fac + 9 * self.b_fac ** 2))

        # Calculations for compressibility

        self.beta = self.V0 * (self.B - self.a2 * self.y ** 2) / (
                    self.vol * (self.B + self.a1 * self.y + self.a2 * self.y ** 2) ** 2)
        self.comp = un.atm_2_bar(self.beta)

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
            Dielectric constant according to Archer and Wang :cite:`Archer1990`

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
            Osmotic coefficient unavailable for this model
            """
        pass

    def a_v(self):
        """
            Apparent molal volume unavailable for this model
            """
        pass

    def a_h(self):
        """
            Enthalpy coefficient unavailable for this model
            """
        pass
