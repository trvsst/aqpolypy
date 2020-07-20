"""
:module: SaltMgSO4PP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Phutela & Pitzer model calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added MgSO4 parameters from Phutela & Pitzer
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.salt.SaltModelPitzer as rp


class MgSO4PropertiesPhutelaPitzer(rp.SaltPropertiesPitzer):
    """
    MgSO4 properties following the work of Phutela and Pitzer :cite:`Phutela1986`

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

        # Calculations MgSO4 parameters and coefficients
        self.mat_stoich = np.array([[1, 1], [2, -2]])

        self.m_ref = 5.550825
        self.y_ref = 10
        self.m_weight = 120.366
        self.p_ref = np.array([self.m_weight, self.m_ref, self.y_ref])

        # values for ion strength dependence and ion size constants in the extended Pitzer model
        self.alpha_b1 = 1.4
        self.alpha_b2 = 12.0
        self.alpha_c1 = 0
        self.alpha_c2 = 0
        self.alpha_d1 = 0
        self.alpha_d2 = 0
        self.b_param = 1.2
        self.ion_param = np.array([self.alpha_b1, self.alpha_b2, self.alpha_c1, self.alpha_c2, self.alpha_d1, self.alpha_d2, self.b_param])

        self.q = np.zeros(15)

        self.q[0] = -7.97433e5
        self.q[1] = 8.94735e3
        self.q[2] = -3.78219e1
        self.q[3] = 7.15424e-2
        self.q[4] = -5.14523e-5
        self.q[5] = 7.42784
        self.q[6] = -6.12133e-2
        self.q[7] = 1.77378e-4
        self.q[8] = -1.82737e-7
        self.q[9] = 2.06777e3
        self.q[10] = -2.39043e1
        self.q[11] = 1.03110e-1
        self.q[12] = -1.96675e-4
        self.q[13] = 1.40527e-7
        self.q[14] = 1.18481e-4

        # Pitzer Parameters
        self.beta0 = 0
        self.beta1 = 0
        self.beta2 = 0
        self.C0 = 0
        self.C1 = 0
        self.C2 = 0
        self.D0 = 0
        self.D1 = 0
        self.D2 = 0

        self.params = np.array([self.beta0, self.beta1, self.beta2, self.C0, self.C1, self.C2, self.D0, self.D1, self.D2])

        # Pitzer Parameters pressure derivative
        self.pr = un.atm_2_bar(self.pa)

        self.vp = self.q[0] / self.tk + self.q[1] + self.q[2] * self.tk + self.q[3] * self.tk ** 2 + self.q[4] * self.tk ** 3
        self.beta0_der_p = 0
        self.beta1_der_p = self.q[5] / self.tk + self.q[6] + self.q[7] * self.tk + self.q[8] * self.tk ** 2
        self.beta2_der_p = self.q[9] / self.tk + self.q[10] + self.q[11] * self.tk + self.q[12] * self.tk ** 2 + self.q[13] * self.tk ** 3 + self.q[14] * self.pr
        self.C0_der_p = 0
        self.C1_der_p = 0
        self.C2_der_p = 0
        self.D0_der_p = 0
        self.D1_der_p = 0
        self.D2_der_p = 0

        self.params_der_p = np.array([self.vp, self.beta0_der_p, self.beta1_der_p, self.beta2_der_p, self.C0_der_p, self.C1_der_p, self.C2_der_p, self.D0_der_p, self.D1_der_p, self.D2_der_p])

        # Pitzer Parameters temperature derivative
        self.beta0_der_t = 0
        self.beta1_der_t = 0
        self.beta2_der_t = 0
        self.C0_der_t = 0
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

        :return: fitting coefficients for MgSO4 (list)

        """
        return [self.mat_stoich, self.p_ref]

    def pitzer_parameters(self):
        """
        returns the values of the Pitzer Parameters as a list

        :return: Pitzer Parameters for MgSO4 (array)

        """
        return self.params

    def pitzer_parameters_der_p(self):
        """
        returns the values of the Pitzer Parameters pressure derivative as a list

        :return: Pitzer Parameters pressure derivative for MgSO4 (array)

        """
        return self.params_der_p

    def pitzer_parameters_der_t(self):
        """
        returns the values of the Pitzer Parameters temperature derivative as a list

        :return: Pitzer Parameters temperature derivative for MgSO4 (array)

        """
        return self.params_der_t

    def ion_parameters(self):
        """
        returns the values of the ionic strength dependence (alpha) & ion-size (b) parameters as a list

        :return: ionic strength dependence & ion-size parameters for MgSO4 (array)

        """
        return self.ion_param
