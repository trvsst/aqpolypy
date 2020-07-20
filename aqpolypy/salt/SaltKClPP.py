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

        # values for ion strength dependence and ion size constants in the extended Pitzer model
        self.alpha_b1 = 2.0
        self.alpha_b2 = 0
        self.alpha_c1 = 0
        self.alpha_c2 = 0
        self.alpha_d1 = 0
        self.alpha_d2 = 0
        self.b_param = 1.2
        self.ion_param = np.array([self.alpha_b1, self.alpha_b2, self.alpha_c1, self.alpha_c2, self.alpha_d1, self.alpha_d2, self.b_param])

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

        self.u = np.array([[-2.10289e-2, 2.208130e-1, 0.000000000],
                           [6.039670e-1, -4.61849000, 7.648910e-4],
                           [3.677680e-3, -4.10116e-2, 0.000000000],
                           [-7.05537e-6, 1.104450e-4, -1.12131e-8],
                           [1.979680e-9, -4.73196e-8, 1.72256e-11],
                           [-2.47588e-3, -2.74120e-2, 0.000000000],
                           [1.441600e-1, 3.328830e-1, -5.71188e-3]])

        self.fl_1 = [6.771360e-4, 9.678540e-4, -4.12364e-5]
        self.fg_1 = [4.808000e-2, 2.187520e-1, -3.94000e-4]
        self.fl = [6.56838000e-4, 9.678540e-4, -4.12364e-5]
        self.fg = [5.00380000e-2, 2.187520e-1, -3.94000e-4]
        self.k_1 = [-2931.268116, 6353.355434, 28.17218000]
        self.k_2 = [-33.95314300, 193.0040590, -0.12556700]

        # Pitzer Parameters
        self.tc = 298.15

        def Fg(x):
            f_1 = (self.u[0][x] * self.tk ** 2) / 6 + (self.u[1][x] * self.tk) / 2 + (self.u[2][x] * self.tk ** 2) * ((np.log(self.tk) / 2) - (5 / 12)) / 3
            f_2 = (self.u[3][x] * self.tk ** 3) / 12 + (self.u[4][x] * self.tk ** 4) / 20 + self.u[5][x] * (self.tk / 2 + (3 * (227 ** 2)) / (2 * self.tk) + 227 * (self.tk - 227) * np.log(self.tk - 227) / self.tk)
            f_3 = (-self.u[6][x]) * (2 * (647 - self.tk) * np.log(647 - self.tk) / self.tk + np.log(647 - self.tk))
            f_4 = (-self.k_1[x]) / self.tk - self.fl[x] * ((self.tc ** 2) / self.tk) + self.k_2[x] + self.fg[x]
            f_sum = f_1 + f_2 + f_3 + f_4
            return f_sum

        self.beta0 = Fg(0)
        self.beta1 = Fg(1)
        self.beta2 = 0
        self.C0 = Fg(2)
        self.C1 = 0
        self.C2 = 0
        self.D0 = 0
        self.D1 = 0
        self.D2 = 0

        self.params = np.array([self.beta0, self.beta1, self.beta2, self.C0, self.C1, self.C2, self.D0, self.D1, self.D2])

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

        self.beta0_der_t = Fl(0)
        self.beta1_der_t = Fl(1)
        self.beta2_der_t = 0
        self.C0_der_t = Fl(2)
        self.C1_der_t = 0
        self.C2_der_t = 0
        self.D0_der_t = 0
        self.D1_der_t = 0
        self.D2_der_t = 0
        self.params_der_t = np.array([self.beta0_der_t, self.beta1_der_t, self.beta2_der_t, self.C0_der_t, self.C1_der_t, self.C2_der_t, self.D0_der_t, self.D1_der_t, self.D2_der_t])

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

    def ion_parameters(self):
        return self.ion_param
