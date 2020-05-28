import numpy as np
from units.Units import *
from water.WaterPropertiesABC import *


class Water01(WaterProperties):

    def __init__(self):
        super().__init__()

    def density(self, T, p=1, compute_beta=False):
        """
            Water density according to Fine Millero
            Journal of Chemical Physics 59, 5529 (1973)

            restricted to temperatures in range [0, 100]

            :param T: absolute temperature
            :param p: pressure in atmosphere
            :return: water density in SI
            :rtype: float


            TODO: include warning if the temperature is outside [0,100]
            """
        if (T > 100 or T < 0):
            return print("Temperature out of allowable range: [0, 100]")

        t = T - celsius_2_kelvin()
        y = atm_2_bar() * (p - 1)

        den1 = 0.9998396 + t * (18.224944e-3) - (7.922210e-6) * t ** 2 - (55.44846e-9) * t ** 3
        den2 = (149.7562e-12) * t ** 4 - (393.2952e-15) * t ** 5

        V0 = (1 + t * 18.159725e-3) / (den1 + den2)

        B = 19654.320 + 147.037 * t - 2.21554 * t ** 2 + (1.0478e-2) * t ** 3 - (2.2789e-5) * t ** 4

        a1 = 3.2891 - (2.3910e-3) * t + (2.8446e-4) * t ** 2 - (2.82e-6) * t ** 3 + (8.477e-9) * t ** 4
        a2 = 6.245e-5 - (3.913e-6) * t - (3.499e-8) * t ** 2 + (7.942e-10) * t ** 3 - (3.299e-12) * t ** 4

        # this in cm^3/gram
        vol = V0 * (1 - y / (B + a1 * y + a2 * y ** 2))

        # density in kg/m^3
        rt = 1e3 / vol

        if compute_beta:
            beta = V0 * (B - a2 * y ** 2) / (vol * (B + a1 * y + a2 * y ** 2) ** 2)
            rt = beta * atm_2_bar()

        return rt

    def molar_volume(self, T, p=1):
        """
            molar water according to Fine Millero
            Journal of Chemical Physics 59, 5529 (1973)

            restricted to temperatures in range [0, 100]

            :param T: absolute temperature
            :param p: pressure in atmosphere
            :return: water density in SI
            :rtype: float


            TODO: include warning if the temperature is outside [0,100]
            """
        if (T > 100 or T < 0):
            return print("Temperature out of allowable range: [0, 100]")

        return 1e-3 * self.MolecularWeight / self.density(T, p)

    def dielectric_constant(self, T, p=1):
        """
            Water dielectric constant according to

            Archer & Wang
            Journal of Physical and Chemical Reference Data 19, 371 (1990)

            :param T: absolute temperature
            :param P: Pressure (in atm)
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
        p_mpa = 1e-6 * atm_2_pascal() * p

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
            t2 = b[4] / (T - 215) ** 0.25
            t3 = np.exp(b[5] / x + b[6] / x ** 2 + b[7] * y / x + b[8] * y / x ** 2)
            return 1 + (1e-3) * self.density(x, p) * (t1 + t2 + t3)

        fac = one_over4pi_epsilon0() * 4 * np.pi * mu ** 2 / (3 * k_boltzmann() * T)
        b_fac_0 = fac * g(T, p_mpa)
        b_fac_1 = alpha + b_fac_0
        b_fac = self.density(T, p) * avogadro() * b_fac_1 / (3.0 * m_w)

        return 0.25 * (1 + 9 * b_fac + 3 * np.sqrt(1 + 2 * b_fac + 9 * b_fac ** 2))

    def compressibility(self):
        pass
