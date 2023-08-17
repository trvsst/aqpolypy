"""
:module: SaltNaClRP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Rogers & Pitzer model calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, Septmber2023
.. history:
..
"""

import numpy as np
import pandas as pd
import importlib.resources
import pathlib

import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as bp
import aqpolypy.salt.SaltModelPitzer as rp


class SaltPropertiesPitzerMayorga(rp.SaltPropertiesPitzer):
    """
    NaCl properties following the work of Rogers and Pitzer :cite:`Rogers1982`

    """

    def __init__(self, salt_type, pa=1):
        """
        constructor

        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, stoichiometry coefficients, Pitzer Parameters

        """
        self.tk = 298.15
        self.pa = pa
        self.salt_type=salt_type

        # Calculations nacl parameters and coefficients
        #self.m_ref = 5.550825
        self.m_ref = None
        #self.y_ref = 10
        self.y_ref = None
        #self.m_weight = 58.4428
        self.m_weight = None
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

        str_n = 'paramsPM.xlsx'
        file_in_2 = pathlib.Path(__file__).parent.joinpath(str_n)
        file_in = importlib.resources.path(__package__, str_n)

        df = pd.read_excel(file_in_2)

        indx = np.where(df['compound']==self.salt_type)
        value = df.iloc[indx[0][0]]
        self.mat_stoich = np.array([[value['np'], value['nm']], [value['zp'], value['zm']]])

        self.beta0 = value['b0']
        self.beta1 = value['b1']
        self.beta2 = 0.0

        self.C0 = 0.5*value['c']
        self.C1 = 0
        self.C2 = 0
        self.D0 = 0
        self.D1 = 0
        self.D2 = 0
        self.params = np.array([self.beta0, self.beta1, self.beta2, self.C0, self.C1, self.C2, self.D0, self.D1, self.D2])

        # we do not need derivatives of the programs
        self.params_der_p = None
        self.params_der_t = None

        super().__init__(self.tk, pa)

    def actual_coefficients(self):
        """
        returns the values of the coefficients as a list

        :return: fitting coefficients for NaCl (list)

        """
        return [self.mat_stoich, self.p_ref]

    def pitzer_parameters(self):
        """
        returns the values of the Pitzer Parameters as a list

        :return: Pitzer Parameters for NaCl (array)

        """
        return self.params

    def pitzer_parameters_der_p(self):
        """
        returns the values of the Pitzer Parameters pressure derivative as a list

        :return: Pitzer Parameters pressure derivative for NaCl (array)

        """
        return self.params_der_p

    def pitzer_parameters_der_t(self):
        """
        returns the values of the Pitzer Parameters temperature derivative as a list

        :return: Pitzer Parameters temperature derivative for NaCl (array)

        """

        return self.params_der_t

    def ion_parameters(self):
        """
        returns the values of the ionic strength dependence (alpha) & ion-size (b) parameters as a list

        :return: ionic strength dependence & ion-size parameters for NaCl (array)

        """
        return self.ion_param
