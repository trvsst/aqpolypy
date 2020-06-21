"""
:module: SaltModelPitzer
:platform: Unix, Windows, OS
:synopsis: Implements Rogers & Pitzer model calculations to SaltPropertiesABC

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Implemented member methods
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as wp
import aqpolypy.salt.SaltPropertiesABC as sp
from abc import ABC, abstractmethod


class SaltPropertiesPitzer(sp.SaltProperties, ABC):
    """
    Salt properties following the work of Pitzer :cite:`Pitzer1973a`

    """

    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, stoichiometry coefficients

        """

        super().__init__(tk, pa)

        # electrolyte un-instantiated parameters
        self.mat_stoich, self.cm, self.p_ref, self.qm = self.actual_coefficients()

        # calculations for params
        self.pr = self.pa * un.atm_2_bar(1)
        self.pr_atm = un.atm_2_bar(1)

        self.vp_0 = self.cm[0] + self.cm[1] * self.tk + self.cm[2] * self.tk ** 2 + self.cm[3] * self.tk ** 3
        self.vp_1 = (self.pr - self.pr_atm) * (self.cm[4] + self.cm[5] * self.tk + self.cm[6] * self.tk ** 2)
        self.vp_2 = (self.pr - self.pr_atm) ** 2 * (self.cm[7] + self.cm[8] * self.tk)

        self.vp = self.vp_0 + self.vp_1 + self.vp_2

        self.bp_0 = self.cm[9] + self.cm[10] / (self.tk - 227) + self.cm[11] * self.tk + self.cm[12] * self.tk ** 2 + self.cm[13] / (680 - self.tk)
        self.bp_1_1 = self.cm[14] + self.cm[15] / (self.tk - 227) + self.cm[16] * self.tk + self.cm[17] * self.tk ** 2
        self.bp_1 = (self.bp_1_1 + self.cm[18] / (680 - self.tk)) * (self.pr - self.pr_atm)
        self.bp_2_1 = self.cm[19] + self.cm[20] / (self.tk - 227) + self.cm[21] * self.tk + self.cm[22] / (680 - self.tk)
        self.bp_2 = self.bp_2_1 * (self.pr - self.pr_atm) ** 2

        self.bp = self.bp_0 + self.bp_1 + self.bp_2

        self.cq = self.cm[23] + self.cm[24] / (self.tk - 227) + self.cm[25] * self.tk + self.cm[26] * self.tk ** 2 + self.cm[27] / (680 - self.tk)
        self.cp = 0.5 * self.cq
        self.params = np.array([self.vp, self.bp, self.cp])

        # calculations for stoichiometry coefficients
        # nu_+ + nu_-
        self.nu = np.sum(self.mat_stoich[0])
        # nu_+ nu_-
        self.nu_prod = self.mat_stoich[0, 0] * self.mat_stoich[0, 1]
        # abs(z_+ z_-)
        self.z_prod = np.abs(self.mat_stoich[1, 0] * self.mat_stoich[1, 1])
        # nu_+, z_+
        self.nz_prod_plus = self.mat_stoich[0, 0] * self.mat_stoich[1, 0]

        self.mat = np.array([self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus])

    @abstractmethod
    def actual_coefficients(self):
        pass

    @staticmethod
    def h_fun(i_str):
        """
            Function for apparent molal volume according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: pitzer_function_1

                f_{p1}(I)=\\frac{1}{2b}\\ln\\left(1+bI^{\\frac{1}{2}}\\right)

            :return: value of function  (float)
            """
        # units are (kg/mol)^{1/2}
        b_param = 1.2
        h_fun = 0.5 * np.log(1 + b_param * np.sqrt(i_str)) / b_param

        return h_fun

    def h_fun_gamma(self, i_str):
        """
             Function in activity coefficient according to Pitzer :cite:`Pitzer1973a`

             .. math::
                :label: pitzer_function_2

                f_{p2}(I)=\\frac{I^{\\frac{1}{2}}}{1+ bI^{\\frac{1}{2}}}+ 4f_{p1}(I)

            :math:`f_{p1}(I)` is defined in :eq:`pitzer_function_1`

            :return: value of function (float)
            """
        b_param = 1.2
        h_fun_gamma = 4 * self.h_fun(i_str) + np.sqrt(i_str) / (1 + b_param * np.sqrt(i_str))

        return h_fun_gamma

    @staticmethod
    def p_fun_gamma(i_str):
        """
            function in activity coefficient according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: pitzer_function_3

                f_{p3}(I)=\\frac{1}{\\alpha^2I}\\left[1-\\left(1+\\alpha I^{\\frac{1}{2}}
                -\\frac{\\alpha^2I}{2}\\right)e^{-\\alpha I^{\\frac{1}{2}}}\\right]

            :return: value of function (float)
            """
        # units are (kg/mol)^{1/2}
        alpha = 2.0

        x = alpha * np.sqrt(i_str)

        p_fun_gamma = (1 - (1 + x - 0.5 * x ** 2) * np.exp(-x)) / x ** 2

        return p_fun_gamma

    def ionic_strength(self, m):
        """
            Ionic strength

            :return: ionic strength in SI (float)
            """
        i_str = 0.5 * m * np.sum(self.mat_stoich[0] * self.mat_stoich[1] ** 2)

        return i_str

    def molar_vol_infinite_dilution(self):
        """
            Partial molal volume of solute at infinite dilution according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: mol_infty_dilution

                `\\bar{\\upsilon}^{\\circ}_s = \\frac{V_{(m_1)}}{m_1} - \\upsilon_wY-\\nu|z_Mz_X|A_vh_{(I)}
                - 2\\nu_M\\nu_XRT\\left(m_1B^{v}_{MX}+\\left(\\nu_Mz_M\\right)m^2_1C^{v}_{MX}\\right)`

            :return: Partial molal volume of solute at infinite dilution in SI (float)
            """
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # molar volume water in cm^3/mol
        vol_water = 1e6 * wp.WaterPropertiesFineMillero(self.tk, self.pa).molar_volume()

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat
        m_r = self.p_ref[1]
        y_r = self.p_ref[2]

        vp, bp, cp = self.params

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
        """
            Density of electrolyte solution according to Pitzer :cite:`Pitzer1973a`

            :return: Density of electrolyte solution in SI (float)
            """
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
        """
            Molar volume of electrolyte solution according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: molar_vol

                \\upsilon_s = \\bar{\\upsilon}^{\\circ}_s+\\nu|z_Mz_X|A_vh_{(I)}+2\\nu_M\\nu_XRT\\left(mB^{v}_{MX}
                +\\left(\\nu_Mz_M\\right)m^2C^{v}_{MX}\\right)

            :return: Molar volume of electrolyte solution in SI (float)
            """
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # coefficients V_m, B, C
        vp, bp, cp = self.params

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
        """
            Osmotic coefficient according to Pitzer :cite:`Pitzer1973a`

            .. math::
                \\phi = 1 - |z_Mz_X|A_\\phi\\frac{I^{\\frac{1}{2}}}{1+bI^{\\frac{1}{2}}}+m\\frac{2\\nu_M\\nu_X}{\\nu}
                \\left(\\beta^{(0)}_{MX}\\beta^{(1)}_{MX}e^{-\\alpha I^{\\frac{1}{2}}}\\right)+m^{2}
                \\frac{2\\left(\\nu_M\\nu_X\\right)^{\\frac{3}{2}}}{\\nu}C^{\\phi}_{MX}

            :return: osmotic coefficient in SI (float)
            """
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
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # ionic strength
        i_str = self.ionic_strength(m)

        x = np.sqrt(i_str)

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, press).a_phi() * x / (1 + 1.2 * x)
        val_2 = 2 * m * (nu_prod / nu) * (beta0 + beta1 * np.exp(-2 * x))
        val_3 = (2 * nu_prod ** 1.5 / nu) * m ** 2 * c_phi

        osmotic_coefficient = 1 + val_1 + val_2 + val_3

        return osmotic_coefficient

    def log_gamma(self, m):
        """
            Activity coefficient according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: pitzer_activity

                \\ln\\gamma_{\\pm} = -|z_Mz_X|A_{\\phi}f_{p2}(I)+m\\frac{2\\nu_M\\nu_X}{\\nu}
                \\left(2\\beta^{(0)}_{MX}+2\\beta^{(1)}_{MX}f_{p3}(I)\\right)+\\frac{3m^2}{2}
                \\left(\\frac{2\\left(\\nu_M\\nu_X\\right)}{\\nu}C^{\\phi}_{MX}\\right)`

            functions are defined in Eq. :eq:`pitzer_function_2`, :eq:`pitzer_function_3`

            :return: activity coefficient (float)
            """
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
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # ionic strength
        i_str = self.ionic_strength(m)

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, press).a_phi() * self.h_fun_gamma(i_str)
        val_2 = 2 * m * (nu_prod / nu) * (2 * beta0 + 2 * beta1 * self.p_fun_gamma(i_str))
        val_3 = 1.5 * (2 * nu_prod ** 1.5 / nu) * m ** 2 * c_phi

        log_gamma = val_1 + val_2 + val_3

        return log_gamma
