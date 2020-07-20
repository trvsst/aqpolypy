"""
:module: SaltNa2SO4PP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Phutela & Pitzer model calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added Na2SO4 parameters from Phutela & Pitzer
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.salt.SaltModelPitzer as rp


class Na2SO4PropertiesPhutelaPitzer(rp.SaltPropertiesPitzer):
    """
    Na2SO4 properties following the work of Phutela and Pitzer :cite:`Phutela1986`

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

        # Calculations Na2SO4 parameters and coefficients
        self.mat_stoich = np.array([[2, 1], [1, -2]])

        self.m_ref = 5.550825
        self.y_ref = 10
        self.m_weight = 142.04
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

        self.p = np.zeros(12)

        self.p[0] = -5.11686e5
        self.p[1] = 5.43000e3
        self.p[2] = -2.18218e1
        self.p[3] = 3.99915e-2
        self.p[4] = -2.83513e-5
        self.p[5] = 1.79274e-2
        self.p[6] = 1.28610
        self.p[7] = -1.06978e-2
        self.p[8] = 2.96022e-5
        self.p[9] = -2.70365e-8
        self.p[10] = 4.59690e-1
        self.p[11] = -1.31097e-3

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

        self.vp = self.p[0] / self.tk + self.p[1] + self.p[2] * self.tk + self.p[3] * self.tk ** 2 + self.p[4] * self.tk ** 3 + self.p[5] * self.pr
        self.beta0_der_p = self.p[6] / self.tk + self.p[7] + self.p[8] * self.tk + self.p[9] * self.tk ** 2
        self.beta1_der_p = self.p[10] / self.tk + self.p[11]
        self.beta2_der_p = 0
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

        :return: fitting coefficients for Na2SO4 (list)

        """
        return [self.mat_stoich, self.p_ref]

    def pitzer_parameters(self):
        """
        returns the values of the Pitzer Parameters as a list

        :return: Pitzer Parameters for Na2SO4 (array)

        """
        return self.params

    def pitzer_parameters_der_p(self):
        """
        returns the values of the Pitzer Parameters pressure derivative as a list

        :return: Pitzer Parameters pressure derivative for Na2SO4 (array)

        """
        return self.params_der_p

    def pitzer_parameters_der_t(self):
        """
        returns the values of the Pitzer Parameters temperature derivative as a list

        :return: Pitzer Parameters temperature derivative for Na2SO4 (array)

        """
        return self.params_der_t

    def ion_parameters(self):
        """
        returns the values of the ionic strength dependence (alpha) & ion-size (b) parameters as a list

        :return: ionic strength dependence & ion-size parameters for Na2SO4 (array)

        """
        return self.ion_param
