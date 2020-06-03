"""
:module: WaterPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added constructor with temperature and pressure parameters.
..                  - Updated member methods to use attributes and functions from Units.
"""

import numpy as np
import aqpolypy.units.Units as un
import aqpolypy.water.WaterPropertiesABC as wp


class WaterPropertiesFineMillero(wp.WaterProperties):

    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: molecular weight, temperature, and pressure
        :itype : float
        """
        self.MolecularWeight = 18.01534
        self.tk = tk
        self.pa = pa

    def density(self):
        """
            Water density according to Fine Millero
            Journal of Chemical Physics 59, 5529 (1973)

            restricted to temperatures in range [0, 100]

            :param tk: absolute temperature
            :param pa: pressure in atmosphere
            :return: water density in SI
            :rtype: float


            TODO: include warning if the temperature is outside [0,100]
            """
        # if (self.tk > 100 or self.tk < 0):
        #    return print("Temperature out of allowable range: [0, 100]")

        t = self.tk - un.celsius_2_kelvin(0)
        y = un.atm_2_bar(self.pa) * (self.pa - 1)

        den1 = 0.9998396 + t * 18.224944e-3 - 7.922210e-6 * t ** 2 - 55.44846e-9 * t ** 3
        den2 = 149.7562e-12 * t ** 4 - 393.2952e-15 * t ** 5

        V0 = (1 + t * 18.159725e-3) / (den1 + den2)

        B = 19654.320 + 147.037 * t - 2.21554 * t ** 2 + 1.0478e-2 * t ** 3 - 2.2789e-5 * t ** 4

        a1 = 3.2891 - 2.3910e-3 * t + 2.8446e-4 * t ** 2 - 2.82e-6 * t ** 3 + 8.477e-9 * t ** 4
        a2 = 6.245e-5 - 3.913e-6 * t - 3.499e-8 * t ** 2 + 7.942e-10 * t ** 3 - 3.299e-12 * t ** 4

        # this in cm^3/gram
        vol = V0 * (1 - y / (B + a1 * y + a2 * y ** 2))

        # density in kg/m^3
        rt = 1e3 / vol

        if self.compressibility():
            beta = V0 * (B - a2 * y ** 2) / (vol * (B + a1 * y + a2 * y ** 2) ** 2)
            rt = beta * un.atm_2_bar(self.pa)

        return rt

    def molar_volume(self):
        """
            molar water according to Fine Millero
            Journal of Chemical Physics 59, 5529 (1973)

            restricted to temperatures in range [0, 100]

            :param tk: absolute temperature
            :return: water density in SI
            :rtype: float


            TODO: include warning if the temperature is outside [0,100]
            """
        # if (T > 100 or T < 0):
        #    return print("Temperature out of allowable range: [0, 100]")

        return 1e-3 * self.MolecularWeight / self.density()

    def dielectric_constant(self):
        """
            Water dielectric constant according to

            Archer & Wang
            Journal of Physical and Chemical Reference Data 19, 371 (1990)

            :param tk: absolute temperature
            :param pa: Pressure (in atm)
            :return: dielectric constant
            :rtype: float
            """

        # water polarizability
        alpha = 18.1458392e-30
        # dipole water magntidue
        mu = 6.1375776e-30
        # water molecular weight
        m_w = 0.01801528

        # convert from atmospheres to MPa
        p_mpa = 1e6 * un.atm_2_pascal(self.pa)

        # coefficients
        b = np.zeros(9)
        b[0] = -4.044525e-2
        b[1] = 103.6180
        b[2] = 75.32165
        b[3] = -23.23778
        b[4] = -3.548184
        b[5] = -1246.311
        b[6] = 263307.7
        b[7] = -6.928953e-1
        b[8] = -204.4473

        def g(x, y):
            t1 = b[0] * y / x + b[1] / np.sqrt(x) + b[2] / (x - 215) + b[3] / np.sqrt(x - 215)
            t2 = b[4] / (self.tk - 215) ** 0.25
            t3 = np.exp(b[5] / x + b[6] / x ** 2 + b[7] * y / x + b[8] * y / x ** 2)
            return 1 + 1e-3 * self.density() * (t1 + t2 + t3)

        fac = un.one_over4pi_epsilon0() * 4 * np.pi * mu ** 2 / (3 * un.k_boltzmann() * self.tk)
        b_fac_0 = fac * g(self.tk, p_mpa)
        b_fac_1 = alpha + b_fac_0
        b_fac = self.density() * un.avogadro() * b_fac_1 / (3.0 * m_w)

        return 0.25 * (1 + 9 * b_fac + 3 * np.sqrt(1 + 2 * b_fac + 9 * b_fac ** 2))

    def compressibility(self):
        compute_beta = True
        return compute_beta
