"""
:module: SaltKClPP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Pabalan & Pitzer model calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added KCl parameters from Pabalan & Pitzer
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.salt.SaltModelPitzer as rp


class KClPropertiesPabalanPitzer(rp.SaltPropertiesPitzer):
    """
    KCl properties following the work of Pabalan and Pitzer :cite:``

    """

    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, stoichiometry coefficients, Pitzer Parameters

        """
        self.tk = tk
        self.pa = pa

        # Calculations nacl parameters and coefficients
        self.mat_stoich = np.array([[1, 1], [1, -1]])

        self.m_ref = 5.550825
        self.y_ref = 10
        self.m_weight = 74.5513
        self.p_ref = np.array([self.m_weight, self.m_ref, self.y_ref])

        self.q = np.array([[1.56152e3, 0.0],
                           [-1.69234e5, 0.0],
                           [-4.29918, 9.45015e-8],
                           [4.59233e-3, -2.90741e-10],
                           [-3.25686e4, 3.26205e-3],
                           [-6.86887, 8.39662e-7],
                           [7.35220e2, 0.0],
                           [2.02245e-2, -4.41638e-9],
                           [-2.15779e-5, 6.71235e-12],
                           [1.03212e2, -4.42327e-5],
                           [5.34941e-3, -7.97437e-10],
                           [-5.73121e-1, 0.0],
                           [-1.57862e-5, 4.12771e-12],
                           [1.66987e-8, -6.24996e-15],
                           [-7.22012e-2, 4.16221e-8]])

        self.u = np.array([[-2.10289e-2, 2.20813e-1, 0.0],
                           [6.03967e-1, -4.61849, 7.64891e-4],
                           [3.67768e-3, -4.10116e-2, 0.0],
                           [-7.05537e-6, 1.10445e-4, -1.12131e-8],
                           [1.97968e-9, -4.73196e-8, 1.72256e-11],
                           [-2.47588e-3, -2.74120e-2, 0.0],
                           [1.44160e-1, 3.32883e-1, -5.71188e-3]])
        self.fl_1 = [6.77136e-4, 9.67854e-4, -4.12364e-5]
        self.fg_1 = [4.8080e-2, 2.18752e-1, -3.94e-4]
        self.fl = [6.56838e-4, 9.67854e-4, -4.12364e-5]
        self.fg = [5.0038e-2, 2.18752e-1, -3.94e-4]
        self.k_1 = [-2931.268116, 6353.355434, 28.172180]
        self.k_2 = [-33.953143, 193.004059, -0.125567]

        # Pitzer Parameters
        self.tc = 298.15

        def Fg(x):
            f_1 = (self.u[0][x] * self.tk ** 2) / 6 + (self.u[1][x] * self.tk) / 2 + (self.u[2][x] * self.tk ** 2) * ((np.log(self.tk) / 2) - (5 / 12)) / 3
            f_2 = (self.u[3][x] * self.tk ** 3) / 12 + (self.u[4][x] * self.tk ** 4) / 20 + self.u[5][x] * (self.tk / 2 + (3 * (227 ** 2)) / (2 * self.tk) + 227 * (self.tk - 227) * np.log(self.tk - 227) / self.tk)
            f_3 = -self.u[6][x] * (2 * (647 - self.tk) * np.log(647 - self.tk) / self.tk + np.log(647 - self.tk))
            f_4 = -self.k_1[x] / self.tk - self.fl[x] * ((self.tc ** 2) / self.tk) + self.k_2[x] + self.fg[x]
            return f_1 + f_2 + f_3 + f_4

        self.beta0 = Fg(0)
        self.beta1 = Fg(1)
        self.C0 = Fg(2)
        self.C1 = 0
        self.C2 = 0
        self.D0 = 0
        self.D1 = 0
        self.D2 = 0

        self.params = np.array([self.beta0, self.beta1, self.C0, self.C1, self.C2, self.D0, self.D1, self.D2])

        # Pitzer Parameters pressure derivative
        self.pr = self.pa * un.atm_2_bar(1)
        self.pr_atm = un.atm_2_bar(1)

        def Fv(x):
            f_1 = self.q[0][x] + self.q[1][x] / self.tk + self.q[2][x] * self.tk + self.q[3][x] * self.tk ** 2 + self.q[4][x] / (647 - self.tk)
            f_2 = self.pr * (self.q[5][x] + self.q[6][x] / self.tk + self.q[7][x] * self.tk + self.q[8][x] * self.tk ** 2 + self.q[9][x] / (647 - self.tk))
            f_3 = (self.pr ** 2) * (self.q[10][x] + self.q[11][x] / self.tk + self.q[12][x] * self.tk + self.q[13][x] * self.tk ** 2 + self.q[14][x] / (647 - self.tk))
            return f_1 + f_2 + f_3

        self.vp = Fv(0)
        self.bp = Fv(1)
        self.cp = 0

        self.params_der_p = np.array([self.vp, self.bp, self.cp])

        # Pitzer Parameters temperature derivative
        def Fl(x):
            f_1 = (self.u[0][x] * self.tk) / 3 + self.u[1][x] / 2 + self.u[2][x] * self.tk * (np.log(self.tk) - (1 / 3)) / 3
            f_2 = (self.u[3][x] * self.tk ** 2) / 4 + (self.u[4][x] * self.tk ** 3) / 5
            f_3 = (self.u[5][x] / (self.tk ** 2)) * (((self.tk - 227) ** 2) / 2 + 454 * (self.tk - 227) + (227 ** 2) * np.log(self.tk - 227))
            f_4 = (self.u[6][x] / (self.tk ** 2)) * (-(647 - self.tk) + 1294 * np.log(647 - self.tk) + (647 ** 2) / (647 - self.tk))
            f_5 = self.k_1[x] / (self.tk ** 2) + self.fl[x] * ((self.tc ** 2) / (self.tk ** 2))
            return f_1 + f_2 + f_3 + f_4 + f_5

        self.beta_0_der_t = Fl(0)
        self.beta_1_der_t = Fl(1)
        self.c_phi_der_t = Fl(2)
        self.params_der_t = np.array([self.beta_0_der_t, self.beta_1_der_t, self.c_phi_der_t])

        super().__init__(tk, pa)

    def actual_coefficients(self):
        """
        returns the values of the coefficients as a list

        :return: fitting coefficients for KCl (list)

        """
        return [self.mat_stoich, self.p_ref]

    def pitzer_parameters(self):
        """
        returns the values of the Pitzer Parameters as a list

        :return: Pitzer Parameters for KCl (array)

        """
        return self.params

    def pitzer_parameters_der_p(self):
        """
        returns the values of the Pitzer Parameters pressure derivative as a list

        :return: Pitzer Parameters pressure derivative for KCl (array)

        """
        return self.params_der_p

    def pitzer_parameters_der_t(self):
        """
        returns the values of the Pitzer Parameters temperature derivative as a list

        :return: Pitzer Parameters temperature derivative for KCl (array)

        """
        return self.params_der_t
