"""
:module: SaltMgCl2WP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Wang & Pitzer model calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added MgCl2 parameters from Wang & Pitzer
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.salt.SaltModelPitzer as rp


class MgCl2PropertiesWangPitzer(rp.SaltPropertiesPitzer):
    """
    MgCl2 properties following the work of Wang and Pitzer :cite:`Wang1998`

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

        # Calculations MgCl2 parameters and coefficients
        self.mat_stoich = np.array([[1, 2], [2, -1]])

        self.m_ref = 5.550825
        self.y_ref = 10
        self.m_weight = 95.211
        self.p_ref = np.array([self.m_weight, self.m_ref, self.y_ref])

        # values for ion strength dependence and ion size constants in the extended Pitzer model
        self.alpha_b1 = 2.0
        self.alpha_b2 = 0
        self.alpha_c1 = 0.4
        self.alpha_c2 = 0.28
        self.alpha_d1 = 0
        self.alpha_d2 = 0
        self.b_param = 1.2
        self.ion_param = np.array([self.alpha_b1, self.alpha_b2, self.alpha_c1, self.alpha_c2, self.alpha_d1, self.alpha_d2, self.b_param])

        self.a = np.array([[[-5.50111455e1, 7.2122055200e1, 5.924282400000, 0.000000000000, 0.000000000000, 4.0898005200e-2],
                            [1.5013032600e1, -1.771450850e1, -1.65126386000, -1.02256042000, 0.000000000000, 0.0000000000000],
                            [-1.58107430e-1, 1.143971530e-1, 1.893998220e-2, 3.770186170e-2, -2.28040769e-3, -2.951198450e-4],
                            [2.304099190e-4, 0.000000000000, -2.99972128e-5, -7.91682934e-5, 1.374258890e-5, 6.9100122700e-7],
                            [-1.31768095e-7, -1.43588435e-7, 1.891742910e-8, 5.913142580e-8, -1.94821902e-8, -5.32314849e-10],
                            [-1.26699609e-28, 1.72952766e-27, 0.00000000000, 0.000000000000, 1.04649784e-28, 3.979618090e-31],
                            [2.821974990e2, 3.41920714000e3, 5.4903020100e1, -2.284930840e2, 0.000000000000, 0.0000000000000]],

                           [[0.00000000000, 0.0000000000000, 4.501140480e-2, 0.000000000000, 0.000000000000, 0.0000000000000],
                            [0.00000000000, 2.2844061200e-4, -1.08427926e-2, 0.000000000000, 0.000000000000, 0.0000000000000],
                            [8.396619600e-5, 0.000000000000, 7.410418640e-5, -7.79259941e-5, 0.000000000000, 0.0000000000000],
                            [-4.60207270e-7, 0.000000000000, -5.99961498e-8, 4.286758760e-7, 0.000000000000, 0.0000000000000],
                            [6.21165614e-10, 0.000000000000, 0.000000000000, -5.77509662e-10, 0.000000000000, 0.000000000000],
                            [8.43555937e-31, -1.77573402e-29, 0.00000000000, 0.000000000000, 0.000000000000, 0.0000000000000],
                            [0.000000000000, -2.2966887900e2, -4.6056284700, 0.000000000000, 0.000000000000, 0.0000000000000]],

                           [[0.000000000000, 0.0000000000000, 0.00000000000, -5.13962051e-4, 0.000000000000, 0.0000000000000],
                            [0.000000000000, 0.0000000000000, 0.00000000000, 9.307611420e-5, 0.000000000000, 0.0000000000000],
                            [0.000000000000, -2.714850860e-7, 0.00000000000, 0.000000000000, 0.000000000000, 0.0000000000000],
                            [0.000000000000, 0.0000000000000, 0.00000000000, 0.000000000000, 0.000000000000, 0.0000000000000],
                            [0.000000000000, 0.0000000000000, -1.39016981e-15, -7.43350922e-13, 0.000000000000, 0.0000000000],
                            [0.000000000000, 0.0000000000000, 0.00000000000, 0.000000000000, 0.000000000000, 0.0000000000000],
                            [-1.11176553000, 1.01000272e1, 1.40556304000e-1, 1.127215570000, 0.000000000000, 0.0000000000000]]])

        self.b = np.array([[-1.17446972e4, 2.865416060e3, -2.279680760e1, 3.09897542e-2, -1.92379975e-5, 1.674471870e-26, -1.91185804e6],
                           [0.00000000000, 0.00000000000, -4.00100143e-4, 0.00000000000, -2.43204343e-8, -6.04921882e-28, 1.080249060e5],
                           [0.00000000000, 1.19091585e-2, -4.19764987e-4, 7.93310194e-7, 0.000000000000, 0.0000000000000, -1.66310231e3]])

        # Pitzer Parameters
        self.pr = un.atm_2_pascal(self.pa) / 1e6
        self.tc = 298.15

        def Fg(x):
            f_1 = self.a[0][0][x] + self.a[0][1][x] * np.log(self.tk) + self.a[0][2][x] * self.tk + self.a[0][3][x] * self.tk ** 2 + self.a[0][4][x] * self.tk ** 3 + self.a[0][5][x] * self.tk ** 10 + self.a[0][6][x] / ((647 - self.tk) ** 2)
            f_2 = self.a[1][0][x] + self.a[1][1][x] * np.log(self.tk) + self.a[1][2][x] * self.tk + self.a[1][3][x] * self.tk ** 2 + self.a[1][4][x] * self.tk ** 3 + self.a[1][5][x] * self.tk ** 10 + self.a[1][6][x] / ((647 - self.tk) ** 2)
            f_3 = self.a[2][0][x] + self.a[2][1][x] * np.log(self.tk) + self.a[2][2][x] * self.tk + self.a[2][3][x] * self.tk ** 2 + self.a[2][4][x] * self.tk ** 3 + self.a[2][5][x] * self.tk ** 10 + self.a[2][6][x] / ((647 - self.tk) ** 2)
            return f_1 + f_2 * self.pr + f_3 * (self.pr ** 2) / 2

        self.beta0 = Fg(0)
        self.beta1 = Fg(1)
        self.beta2 = 0
        self.C0 = Fg(2)
        self.C1 = Fg(3)
        self.C2 = Fg(4)
        self.D0 = Fg(5)
        self.D1 = 0
        self.D2 = 0

        self.params = np.array([self.beta0, self.beta1, self.beta2, self.C0, self.C1, self.C2, self.D0, self.D1, self.D2])

        # Pitzer Parameters pressure derivative
        def Fv(x):
            f_2 = self.a[1][0][x] + self.a[1][1][x] * np.log(self.tk) + self.a[1][2][x] * self.tk + self.a[1][3][x] * self.tk ** 2 + self.a[1][4][x] * self.tk ** 3 + self.a[1][5][x] * self.tk ** 10 + self.a[1][6][x] / ((647 - self.tk) ** 2)
            f_3 = self.a[2][0][x] + self.a[2][1][x] * np.log(self.tk) + self.a[2][2][x] * self.tk + self.a[2][3][x] * self.tk ** 2 + self.a[2][4][x] * self.tk ** 3 + self.a[2][5][x] * self.tk ** 10 + self.a[2][6][x] / ((647 - self.tk) ** 2)
            return f_2 + f_3 * self.pr

        def Fv_vp():
            V_1 = self.b[0][0] + self.b[0][1] * np.log(self.tk) + self.b[0][2] * self.tk + self.b[0][3] * self.tk ** 2 + self.b[0][4] * self.tk ** 3 + self.b[0][5] * self.tk ** 10 + self.b[0][6] / ((647 - self.tk) ** 2)
            V_2 = self.b[1][0] + self.b[1][1] * np.log(self.tk) + self.b[1][2] * self.tk + self.b[1][3] * self.tk ** 2 + self.b[1][4] * self.tk ** 3 + self.b[1][5] * self.tk ** 10 + self.b[1][6] / ((647 - self.tk) ** 2)
            V_3 = self.b[2][0] + self.b[2][1] * np.log(self.tk) + self.b[2][2] * self.tk + self.b[2][3] * self.tk ** 2 + self.b[2][4] * self.tk ** 3 + self.b[2][5] * self.tk ** 10 + self.b[2][6] / ((647 - self.tk) ** 2)
            return V_1 + V_2 * self.pr + V_3 * self.pr ** 2

        self.vp = Fv_vp()
        self.bp = Fv(0)
        self.cp = Fv(2)

        self.params_der_p = np.array([self.vp, self.bp, self.cp])

        # Pitzer Parameters temperature derivative
        def Fl(x):
            f_1 = self.a[0][1][x] / self.tk + self.a[0][2][x] + 2 * self.a[0][3][x] * self.tk + 3 * self.a[0][4][x] * self.tk ** 2 + 10 * self.a[0][5][x] * self.tk ** 9 + 2 * self.a[0][6][x] / ((647 - self.tk) ** 3)
            f_2 = self.a[1][1][x] / self.tk + self.a[1][2][x] + 2 * self.a[1][3][x] * self.tk + 3 * self.a[1][4][x] * self.tk ** 2 + 10 * self.a[1][5][x] * self.tk ** 9 + 2 * self.a[1][6][x] / ((647 - self.tk) ** 3)
            f_3 = self.a[2][1][x] / self.tk + self.a[2][2][x] + 2 * self.a[2][3][x] * self.tk + 3 * self.a[2][4][x] * self.tk ** 2 + 10 * self.a[2][5][x] * self.tk ** 9 + 2 * self.a[2][6][x] / ((647 - self.tk) ** 3)
            return f_1 + f_2 * self.pr + f_3 * (self.pr ** 2) / 2

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

        :return: fitting coefficients for MgCl2 (list)

        """
        return [self.mat_stoich, self.p_ref]

    def pitzer_parameters(self):
        """
        returns the values of the Pitzer Parameters as a list

        :return: Pitzer Parameters for MgCl2 (array)

        """
        return self.params

    def pitzer_parameters_der_p(self):
        """
        returns the values of the Pitzer Parameters pressure derivative as a list

        :return: Pitzer Parameters pressure derivative for MgCl2 (array)

        """
        return self.params_der_p

    def pitzer_parameters_der_t(self):
        """
        returns the values of the Pitzer Parameters temperature derivative as a list

        :return: Pitzer Parameters temperature derivative for MgCl2 (array)

        """
        return self.params_der_t

    def ion_parameters(self):
        return self.ion_param
