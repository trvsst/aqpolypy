"""
:module: SaltGeneralPitzer
:platform: Unix, Windows, OS
:synopsis: Implements Rogers & Pitzer model calculations to SaltPropertiesABC

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Implemented member methods
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as wp
import aqpolypy.salt.SaltPropertiesABC as sp


class SaltPropertiesRogersPitzer(sp.SaltProperties):

    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature absolute
        :param pa: pressure in atm
        :instantiate:

        """

        super().__init__(tk, pa)

        # electrolyte un-instantiated parameters
        self.m_ref = None
        self.y_ref = None
        self.m_weight = None
        self.p_ref = np.array([self.m_weight, self.m_ref, self.y_ref])

        self.cm = np.zeros(28)

        self.qm = np.zeros(19)

    def h_fun(self, i_str):
        # units are (kg/mol)^{1/2}
        b_param = 1.2
        h_fun = 0.5 * np.log(1 + b_param * np.sqrt(i_str)) / b_param

        return h_fun

    def h_fun_gamma(self, i_str):
        b_param = 1.2
        h_fun_gamma = 4 * self.h_fun(i_str) + np.sqrt(i_str) / (1 + b_param * np.sqrt(i_str))
        return h_fun_gamma

    def p_fun_gamma(self, i_str):
        # units are (kg/mol)^{1/2}
        alpha = 2.0

        x = alpha * np.sqrt(i_str)

        p_fun_gamma = (1 - (1 + x - 0.5 * x ** 2) * np.exp(-x)) / x ** 2

        return p_fun_gamma

    def params(self):
        pr = self.pa * un.atm_2_bar(1)
        pr_atm = un.atm_2_bar(1)

        vp_0 = self.cm[0] + self.cm[1] * self.tk + self.cm[2] * self.tk ** 2 + self.cm[3] * self.tk ** 3
        vp_1 = (pr - pr_atm) * (self.cm[4] + self.cm[5] * self.tk + self.cm[6] * self.tk ** 2)
        vp_2 = (pr - pr_atm) ** 2 * (self.cm[7] + self.cm[8] * self.tk)

        vp = vp_0 + vp_1 + vp_2

        bp_0 = self.cm[9] + self.cm[10] / (self.tk - 227) + self.cm[11] * self.tk + self.cm[12] * self.tk ** 2 + self.cm[13] / (680 - self.tk)
        bp_1_1 = self.cm[14] + self.cm[15] / (self.tk - 227) + self.cm[16] * self.tk + self.cm[17] * self.tk ** 2
        bp_1 = (bp_1_1 + self.cm[18] / (680 - self.tk)) * (pr - pr_atm)
        bp_2_1 = self.cm[19] + self.cm[20] / (self.tk - 227) + self.cm[21] * self.tk + self.cm[22] / (680 - self.tk)
        bp_2 = bp_2_1 * (pr - pr_atm) ** 2

        bp = bp_0 + bp_1 + bp_2

        cq = self.cm[23] + self.cm[24] / (self.tk - 227) + self.cm[25] * self.tk + self.cm[26] * self.tk ** 2 + self.cm[27] / (680 - self.tk)
        cp = 0.5 * cq
        params = np.array([vp, bp, cp])

        return params

    def stoichiometry_coeffs(self):
        # nu_+ + nu_-
        nu = np.sum(self.mat_stoich[0])
        # nu_+ nu_-
        nu_prod = self.mat_stoich[0, 0] * self.mat_stoich[0, 1]
        # abs(z_+ z_-)
        z_prod = np.abs(self.mat_stoich[1, 0] * self.mat_stoich[1, 1])
        # nu_+, z_+
        nz_prod_plus = self.mat_stoich[0, 0] * self.mat_stoich[1, 0]

        mat = np.array([nu, nu_prod, z_prod, nz_prod_plus])

        return mat

    def ionic_strength(self, m):
        i_str = 0.5 * m * np.sum(self.mat_stoich[0] * self.mat_stoich[1] ** 2)
        return i_str

    def molar_vol_infinite_dilution(self):
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # molar volume water in cm^3/mol
        vol_water = 1e6 * wp.WaterPropertiesFineMillero(self.tk, self.pa).molar_volume()

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.stoichiometry_coeffs()
        m_r = self.p_ref[1]
        y_r = self.p_ref[2]

        vp, bp, cp = self.params()

        i_str = self.ionic_strength(m_r)

        mv_i_0 = vp / m_r - y_r * vol_water
        mv_i_1 = -nu * z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.h_fun(i_str)
        mv_i_2 = -2 * nu_prod * ct * (m_r * bp + nz_prod_plus * m_r ** 2 * cp)
        # this is in cm^3/mol
        mv_i = mv_i_0 + mv_i_1 + mv_i_2

        # return in SI m^3
        molar_vol_infinite_dilution = 1e-6 * mv_i
        return molar_vol_infinite_dilution

    def density_sol(self, m):
        mw = wp.WaterPropertiesFineMillero(self.tk, self.pa).MolecularWeight
        # convert to cm^3/mol
        v_water = 1e6 * wp.WaterPropertiesFineMillero(self.tk, self.pa).molar_volume()

        # convert to cm^3/mol
        v_salt = 1e6 * self.molar_vol(m)
        mw_salt = self.p_ref[0]

        # density in g/cm^3
        dens = mw * (1 + 1e-3 * m * mw_salt) / (v_water + 1e-3 * m * mw * v_salt)

        # return density in kg/m3
        density_sol = 1e3 * dens
        return density_sol

    def molar_vol(self, m):
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.stoichiometry_coeffs()

        # coefficients V_m, B, C
        vp, bp, cp = self.params()

        # ionic strength
        i_str = self.ionic_strength(m)

        # infinite molar volume, convert to cm^3/mol
        v_1 = 1e6 * self.molar_vol_infinite_dilution()

        val_1 = v_1 + nu * z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.h_fun(i_str)
        val_2 = 2 * nu_prod * ct * (bp * m + nz_prod_plus * cp * m ** 2)

        # molar volume in cm^3/mol
        val = val_1 + val_2

        # return in m^3/mol
        molar_vol = 1e-6 * val
        return molar_vol

    def osmotic_coeff(self, m):
        # pressure is 1 atm
        press = 1
        tc = 298.15

        beta0_1 = self.qm[0] + self.qm[1] * (1 / self.tk - 1 / tc) + self.qm[2] * np.log(self.tk / tc)
        beta0_2 = self.qm[3] * (self.tk - tc) + self.qm[4] * (self.tk ** 2 - tc ** 2)
        beta0 = beta0_1 + beta0_2

        beta1 = self.qm[5] + self.qm[8] * (self.tk - tc) + self.qm[9] * (self.tk ** 2 - tc ** 2)

        c_phi_1 = self.qm[10] + self.qm[11] * (1 / self.tk - 1 / tc) + self.qm[12] * np.log(self.tk / tc)
        c_phi = c_phi_1 + self.qm[13] * (self.tk - tc)

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.stoichiometry_coeffs()

        # ionic strength
        i_str = self.ionic_strength(m)

        x = np.sqrt(i_str)

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, press).a_phi() * x / (1 + 1.2 * x)
        val_2 = 2 * m * (nu_prod / nu) * (beta0 + beta1 * np.exp(-2 * x))
        val_3 = (2 * nu_prod ** 1.5 / nu) * m ** 2 * c_phi

        osmotic_coefficient = 1 + val_1 + val_2 + val_3
        return osmotic_coefficient

    def log_gamma(self, m):
        # pressure is 1 atm
        press = 1
        tc = 298.15

        beta0_1 = self.qm[0] + self.qm[1] * (1 / self.tk - 1 / tc) + self.qm[2] * np.log(self.tk / tc)
        beta0_2 = self.qm[3] * (self.tk - tc) + self.qm[4] * (self.tk ** 2 - tc ** 2)
        beta0 = beta0_1 + beta0_2

        beta1 = self.qm[5] + self.qm[8] * (self.tk - tc) + self.qm[9] * (self.tk ** 2 - tc ** 2)

        c_phi_1 = self.qm[10] + self.qm[11] * (1 / self.tk - 1 / tc) + self.qm[12] * np.log(self.tk / tc)
        c_phi = c_phi_1 + self.qm[13] * (self.tk - tc)

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.stoichiometry_coeffs()

        # ionic strength
        i_str = self.ionic_strength(m)

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, press).a_phi() * self.h_fun_gamma(i_str)
        val_2 = 2 * m * (nu_prod / nu) * (2 * beta0 + 2 * beta1 * self.p_fun_gamma(i_str))
        val_3 = 1.5 * (2 * nu_prod ** 1.5 / nu) * m ** 2 * c_phi

        log_gamma = val_1 + val_2 + val_3
        return log_gamma
