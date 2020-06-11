"""
:module: SaltNaClRP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Rogers Pitzer calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as wp
import aqpolypy.salt.SaltPropertiesABC as sp


class SaltPropertiesRogersPitzer(sp.SaltProperties):
    """
    Slat Properties

    """

    def __init__(self, m, tk, pa=1):
        """
        constructor

        :param :
        :param :
        :instantiate:

        """

        super().__init__(tk, pa)

        """
        Calculations log_gamma_nacl_simple
        """
        self.cf = np.array([1.4495, 2.0442e-2, 5.7927e-3, -2.8860e-4])

        """
        Calculations nacl_params
        """
        self.mat_stoich = np.array([[1, 1], [1, -1]])
        self.m_ref = 5.550825
        self.y_ref = 10
        self.m_weight = 58.4428
        self.p_ref = np.array([self.m_weight, self.m_ref, self.y_ref])

        self.cm = np.zeros(28)

        self.cm[0] = 1.0249125e3
        self.cm[1] = 2.7796679e-1
        self.cm[2] = -3.0203919e-4
        self.cm[3] = 1.4977178e-6
        self.cm[4] = -7.2002329e-2
        self.cm[5] = 3.1453130e-4
        self.cm[6] = -5.9795994e-7
        self.cm[7] = -6.6596010e-6
        self.cm[8] = 3.0407621e-8
        self.cm[9] = 5.3699517e-5
        self.cm[10] = 2.2020163e-3
        self.cm[11] = -2.6538013e-7
        self.cm[12] = 8.6255554e-10
        self.cm[13] = -2.6829310e-2
        self.cm[14] = -1.1173488e-7
        self.cm[15] = -2.6249802e-7
        self.cm[16] = 3.4926500e-10
        self.cm[17] = -8.3571924e-13
        # exponent in 18 (19 in reference) is barely visible, it is 5 (confirmed)
        self.cm[18] = 3.0669940e-5
        self.cm[19] = 1.9767979e-11
        self.cm[20] = -1.9144105e-10
        self.cm[21] = 3.1387857e-14
        self.cm[22] = -9.6461948e-9
        self.cm[23] = 2.2902837e-5
        self.cm[24] = -4.3314252e-4
        self.cm[25] = -9.0550901e-8
        self.cm[26] = 8.6926600e-11
        self.cm[27] = 5.1904777e-4

        self.qm = np.zeros(19)

        self.qm[0] = 0.0765
        self.qm[1] = -777.03
        self.qm[2] = -4.4706
        self.qm[3] = 0.008946
        self.qm[4] = -3.3158e-6
        self.qm[5] = 0.2664
        # this value is not provided
        self.qm[6] = 0
        # this value is not provided
        self.qm[7] = 0
        self.qm[8] = 6.1608e-5
        self.qm[9] = 1.0715e-6
        self.qm[10] = 0.00127
        self.qm[11] = 33.317
        self.qm[12] = 0.09421
        self.qm[13] = -4.655e-5
        # this value is not provided
        self.qm[14] = 0
        self.qm[15] = 41587.11
        self.qm[16] = -315.90
        self.qm[17] = 0.8514
        self.qm[18] = -8.3637e-4

        self.nacl_param = [self.mat_stoich, self.cm, self.p_ref, self.qm]

    def h_fun(self):

        return self.hf

    def h_fun_gamma(self):

        return self.hfg

    def p_fun_gamma(self):
        
        return self.pfg

    def params(self):

        return self.param

    def stoichiometry_coeffs(self):

        return self.mat

    def ionic_strength(self):

        return self.i_str

    def molar_vol_infinite_dilution(self):

        return self.mol_vol_inf_dil

    def density_sol(self):

        return self.dens_sol

    def molar_vol(self):

        return self.mol_vol

    def osmotic_coeff(self):

        return self.osmotic_coefficient

    def log_gamma(self):

        return self.log_g

    def log_gamma_simple(self):

        return self.lgs

    def log_gamma_nacl_simple(self):

        return self.lgs

    def nacl_params(self):

        return self.nacl_param
