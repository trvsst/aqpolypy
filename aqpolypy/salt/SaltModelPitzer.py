"""
:module: SaltModelPitzer
:platform: Unix, Windows, OS
:synopsis: Implements Rogers & Pitzer model calculations to SaltPropertiesABC

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Implemented member methods
..                  - Removed calculations of Pitzer Parameters
..                  - Implemented abstract methods for Pitzer Parameters
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
        :instantiate: temperature, pressure, stoichiometry coefficients, Pitzer Parameters

        """

        super().__init__(tk, pa)

        # electrolyte un-instantiated parameters
        self.mat_stoich, self.p_ref = self.actual_coefficients()

        # Pitzer Parameters
        self.params = self.pitzer_parameters()
        self.params_der_p = self.pitzer_parameters_der_p()
        self.params_der_t = self.pitzer_parameters_der_t()

        # ionic strength dependence (alpha) & ion-size (b) parameters
        self.ion_param = self.ion_parameters()
        self.alpha_b1, self.alpha_b2, self.alpha_c1, self.alpha_c2, self.alpha_d1, self.alpha_d2, self.b_param = self.ion_parameters()

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

    @abstractmethod
    def pitzer_parameters(self):
        pass

    @abstractmethod
    def pitzer_parameters_der_p(self):
        pass

    @abstractmethod
    def pitzer_parameters_der_t(self):
        pass

    @abstractmethod
    def ion_parameters(self):
        pass

    def h_fun(self, i_str):
        """
            Function for apparent molal volume according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: pitzer_function_1

                f_{p1}(I)=\\frac{1}{2b}\\ln\\left(1+bI^{\\frac{1}{2}}\\right)

            :return: value of function  (float)
            """
        # units are (kg/mol)^{1/2}
        b_param = self.b_param
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
        b_param = self.b_param
        h_fun_gamma = 4 * self.h_fun(i_str) + np.sqrt(i_str) / (1 + b_param * np.sqrt(i_str))

        return h_fun_gamma

    @staticmethod
    def p_fun_gamma(a, i_str):
        """
            function in activity coefficient according to Pitzer :cite:`Pitzer1973a`

            .. math::
                :label: pitzer_function_3

                f_{p3}(\\alpha, I)=\\frac{1}{\\alpha^2I}\\left[1-\\left(1+\\alpha I^{\\frac{1}{2}}
                -\\frac{\\alpha^2I}{2}\\right)e^{-\\alpha I^{\\frac{1}{2}}}\\right]

            :return: value of function (float)
            """
        # units are (kg/mol)^{1/2}
        alpha = a

        if a == 0:
            return 0

        x = alpha * np.sqrt(i_str)

        p_fun_gamma = (1 - (1 + x - 0.5 * x ** 2) * np.exp(-x)) / x ** 2

        return p_fun_gamma

    @staticmethod
    def p_fun_gamma_2(a, i_str):
        """
            function in activity coefficient according to Wang & Pitzer:cite:`Wang1998`

            .. math::
                :label: pitzer_function_4

                f_{p4}(\\alpha, I)=\\frac{1}{\\alpha^2I^2}\\left[1-\\left(1+\\alpha I
                - \\alpha^2I^2 \\right)e^{-\\alpha I}\\right]

            :return: value of function (float)
            """
        # units are (kg/mol)^{1/2}
        alpha = a

        if a == 0:
            return 0

        x = alpha * i_str

        p_fun_gamma_2 = (1 - (1 + x - x ** 2) * np.exp(-x)) / x ** 2

        return p_fun_gamma_2

    @staticmethod
    def p_fun_gamma_3(a, i_str):
        """
            function in activity coefficient according to Wang & Pitzer:cite:`Wang1998`

            .. math::
                :label: pitzer_function_5

                f_{p5}(\\alpha, I)=\\frac{1}{\\alpha^2I^3}\\left[1-\\left(1+\\alpha I^{\\frac{3}{2}}
                -\\frac{3\\alpha^2I^3}{2}\\right)e^{-\\alpha I^{\\frac{3}{2}}}\\right]

            :return: value of function (float)
            """
        # units are (kg/mol)^{1/2}
        alpha = a

        if a == 0:
            return 0

        x = alpha * i_str ** 1.5

        p_fun_gamma_2 = (1 - (1 + x - 1.5 * x ** 2) * np.exp(-x)) / x ** 2

        return p_fun_gamma_2

    @staticmethod
    def g_fun_phi(a, i_str):
        """
            function in apparent molal enthalpy according to Wang & Pitzer :cite:`Wang1998`

            .. math::
                :label: wang_function_1

                g_{p1}(x)=\\frac{\\left[1-\\left(1+x\\right)e^{-x}\\right]}{x^{2}}

            :return: value of function (float)
            """

        alpha = a

        if a == 0:
            return 0

        x = alpha * i_str

        g_fun_phi = ((1 - (1 + x)) * np.exp(-x)) / x ** 2

        return g_fun_phi

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

                \\upsilon_1^{\\infty}=\\frac{\\upsilon(m_r)}{m_r}-Y \\upsilon_0^{\\infty}-\\nu|z_{+}z_{-}|A_v h(I_r)
                -2\\nu_+\\nu_{-} RT\\left[ m_r B^{\\nu}_{\\pm}+\\nu_{+}z_{+}m^2_rC^{\\nu}_{\\pm}\\right]

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

        vp, bp, cp = self.params_der_p

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

                \\upsilon_{\\phi}(m)=\\upsilon^{\\infty}_1+\\nu|z_{+}z_{-}|A_v h(I) \\cr +2RT(\\nu_{+}\\nu_{-})m\\left[
                \\frac{\\partial \\beta^{(0)}_{\\pm}}{\\partial P}+2\\frac{\\partial \\beta^{(1)}_{\\pm}}{\\partial P}
                g_{p1}(x)+2\\frac{\\partial \\beta^{(2)}_{\\pm}}{\\partial P}g_{p1}(x)\\right] \\cr
                +2RT(\\nu_{+}\\nu_{-})^{\\frac{3}{2}}m^{2}\\left[\\frac{\\partial C^{(0)}_{\\pm}}{\\partial P}+2\\frac{
                \\partial C^{(1)}_{\\pm}}{\\partial P}g_{p1}(x) + 2\\frac{\\partial C^{(2)}_{\\pm}}{\\partial P}
                g_{p1}(x)\\right] \\cr +2RT(\\nu_{+}\\nu_{-})^2 m^{3}\\left[\\frac{\\partial D^{(0)}_{\\pm}}
                {\\partial P} + 2\\frac{\\partial D^{(1)}_{\\pm}}{\\partial P}g_{p1}(x)) + 2\\frac{\\partial
                D^{(2)}_{\\pm}}{\\partial P}g_{p1}(x)\\right]

            functions are defined in Eq. :eq:`wang_function_1`

            :return: Molar volume of electrolyte solution in SI (float)
            """
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # ionic strength
        i_str = self.ionic_strength(m)

        # Pitzer parameters pressure derivative and Vp
        vp, beta0_der_p, beta1_der_p, beta2_der_p, C0_der_p, C1_der_p, C2_der_p, D0_der_p, D1_der_p, D2_der_p = self.params_der_p

        BV = (beta0_der_p + 2 * beta1_der_p * self.g_fun_phi(self.alpha_b1, i_str ** 0.5) + 2 * beta2_der_p * self.g_fun_phi(self.alpha_b1, i_str ** 0.5))
        CV = (C0_der_p + 2 * C1_der_p * self.g_fun_phi(self.alpha_c1, i_str) + 2 * C2_der_p * self.g_fun_phi(self.alpha_c2, i_str))
        DV = (D0_der_p + 2 * D1_der_p * self.g_fun_phi(self.alpha_d1, i_str ** 1.5) + 2 * D2_der_p * self.g_fun_phi(self.alpha_d2, i_str ** 1.5))

        # infinite molar volume, convert to cm^3/mol
        v_1 = 1e6 * self.molar_vol_infinite_dilution()

        val_1 = v_1 + nu * z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.h_fun(i_str)
        val_2 = 2 * nu_prod * ct * m * BV
        val_3 = 2 * nu_prod ** 1.5 * ct * m ** 2 * CV
        val_4 = 2 * nu_prod ** 2 * ct * m ** 3 * DV

        # molar volume in cm^3/mol
        val = val_1 + val_2 + val_3 + val_4

        # return in m^3/mol
        molar_vol = 1e-6 * val

        return molar_vol

    def osmotic_coeff(self, m):
        """
            Osmotic coefficient according to Wang & Pitzer :cite:`Wang1998`

            .. math::

                \\phi = 1 - |z_{+}z_{-}|A_{\\phi}\\frac{I^{\\frac{1}{2}}}{\\left(1 + bI^{\\frac{1}{2}}\\right)}\\cr
                +\\frac{2\\left(\\nu_{+}\\nu_{-}\\right)}{\\nu}m\\left[\\beta_{\\pm}^{(0)}+\\beta_{\\pm}^{(1)}e^{
                -\\alpha_{B1}I^{\\frac{1}{2}}}+ \\beta_{\\pm}^{(2)}e^{-\\alpha_{B2}I^{\\frac{1}{2}}}\\right] \\cr
                +\\frac{2\\left(\\nu_{+}\\nu_{-}\\right)^{\\frac{3}{2}}}{\\nu}m^{2}\\left[2\\left[C_{\\pm}^{(0)}+
                C_{\\pm}^{(1)}e^{-\\alpha_{C1}I}+ C_{\\pm}^{(2)}e^{-\\alpha_{C2}I}\\right]\\right] \\cr
                +\\frac{2\\left(\\nu_{+}\\nu_{-}\\right)^{2}}{\\nu}m^{3}\\left[3\\left[D_{\\pm}^{(0)}+D_{\\pm}^{(1)}e^{
                -\\alpha_{D1}I^{\\frac{3}{2}}}+ D_{\\pm}^{(2)}e^{-\\alpha_{D2}I^{\\frac{3}{2}}}\\right]\\right] \\cr

            :return: osmotic coefficient in SI (float)
            """
        # pressure is 1 atm
        press = 1

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # ionic strength
        i_str = self.ionic_strength(m)

        # Pitzer Parameters
        beta0, beta1, beta2, C0, C1, C2, D0, D1, D2 = self.params

        B = (beta0 + beta1 * np.exp(-self.alpha_b1 * i_str ** 0.5) + beta2 * np.exp(-self.alpha_b2 * i_str ** 0.5))
        C = (2 * (C0 + C1 * np.exp(-self.alpha_c1 * i_str) + C2 * np.exp(-self.alpha_c2 * i_str)))
        D = (3 * (D0 + D1 * np.exp(-self.alpha_d1 * i_str ** 1.5) + D2 * np.exp(-self.alpha_d2 * i_str ** 1.5)))

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, press).a_phi() * np.sqrt(i_str) / (1 + self.b_param * np.sqrt(i_str))
        val_2 = 2 * m * (nu_prod / nu) * B
        val_3 = (2 * nu_prod ** 1.5 / nu) * m ** 2 * C
        val_4 = (2 * nu_prod ** 2 / nu) * m ** 3 * D

        osmotic_coefficient = 1 + val_1 + val_2 + val_3 + val_4

        return osmotic_coefficient

    def log_gamma(self, m):
        """
            Activity coefficient according to Wang & Pitzer :cite:`Wang1998`

            .. math::
                :label: pitzer_activity

                \\ln \\gamma_{\\pm} = -|z_{+}z_{-}|A_{\\gamma}f_{p2}(I) \\cr
                + \\frac{2(\\nu_{+}\\nu_{-})}{\\nu} m \\left[2\\beta^{(0)}_{\\pm} + 2\\beta^{(1)}_{\\pm}
                f_{p3}(\\alpha_{B1}, I) + 2\\beta^{(2)}_{\\pm}f_{p3}(\\alpha_{B2}, I)\\right]\\cr
                + \\frac{2(\\nu_{+}\\nu_{-})^{\\frac{3}{2}}}{\\nu} m^{2}\\left[3C^{(0)}_{\\pm} + 2C^{(1)}_{\\pm}
                f_{p4}(\\alpha_{C1}, I) + 2C^{(2)}_{\\pm}f_{p4}(\\alpha_{C2}, I)\\right] \\cr
                + \\frac{2(\\nu_{+}\\nu_{-})^{2}}{\\nu} m^{3}\\left[4D^{(0)}_{\\pm} + 2D^{(1)}_{\\pm}
                f_{p5}(\\alpha_{D1}, I) + 2D^{(2)}_{\\pm}f_{p5}(\\alpha_{D2}, I)\\right]

            functions are defined in Eq. :eq:`pitzer_function_2`, :eq:`pitzer_function_3`, :eq:`pitzer_function_4`, :eq:`pitzer_function_5`

            :return: activity coefficient (float)
            """
        # pressure is 1 atm
        press = 1

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # ionic strength
        i_str = self.ionic_strength(m)

        # Pitzer Parameters
        beta0, beta1, beta2, C0, C1, C2, D0, D1, D2 = self.params

        BY = (2 * beta0 + 2 * beta1 * self.p_fun_gamma(self.alpha_b1, i_str) + 2 * beta2 * self.p_fun_gamma(self.alpha_b2, i_str))
        CY = (3 * C0 + 2 * C1 * self.p_fun_gamma_2(self.alpha_c1, i_str) + 2 * C2 * self.p_fun_gamma_2(self.alpha_c2, i_str))
        DY = (4 * D0 + 2 * D1 * self.p_fun_gamma_3(self.alpha_d1, i_str) + 2 * D2 * self.p_fun_gamma_3(self.alpha_d2, i_str))

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, press).a_phi() * self.h_fun_gamma(i_str)
        val_2 = 2 * m * (nu_prod / nu) * BY
        val_3 = (2 * nu_prod ** 1.5 / nu) * m ** 2 * CY
        val_4 = (2 * nu_prod ** 2 / nu) * m ** 3 * DY

        log_gamma = val_1 + val_2 + val_3 + val_4

        return log_gamma

    def apparent_molal_enthalpy(self, m):
        """
            Apparent molal enthalpy according to Wang and Pitzer :cite:`Wang1998`

            .. math::

                L_{\\phi} = \\frac{\\nu|z_{+}z_{-}|A_{H}}{3b} \\ln(1+bI^{\\frac{1}{2}}) \\cr -2RT^{2}(\\nu_{+}
                \\nu_{-})m \\left[\\frac{\\partial \\beta^{(0)}_{\\pm}}{\\partial T}+2\\frac{\\partial
                \\beta^{(1)}_{\\pm}}{\\partial T}g_{p1}(x)+2\\frac{\\partial \\beta^{(2)}_{\\pm}}{\\partial T}
                g_{p1}(x)\\right] \\cr -2RT^{2}(\\nu_{+}\\nu_{-})^{\\frac{3}{2}}m^{2}\\left[\\frac{\\partial
                C^{(0)}_{\\pm}}{\\partial T}+2\\frac{\\partial C^{(1)}_{\\pm}}{\\partial T}g_{p1}(x)+2\\frac{\\partial
                C^{(2)}_{\\pm}}{\\partial T}g_{p1}(x)\\right] \\cr -2RT^{2}(\\nu_{+}\\nu_{-})^{2}m^{3}\\left[
                \\frac{\\partial D^{(0)}_{\\pm}}{\\partial T}+2\\frac{\\partial D^{(1)}_{\\pm}}{\\partial T}g_{p1}(x)
                +2\\frac{\\partial D^{(2)}_{\\pm}}{\\partial T}g_{p1}(x)\\right]

            functions are defined in Eq. :eq:`wang_function_1`

            :return: apparent molal enthalpy in SI (float)
            """
        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # ionic strength
        i_str = self.ionic_strength(m)

        # Pitzer Parameters temperature derivative
        beta0_der_t, beta1_der_t, beta2_der_t, C0_der_t, C1_der_t, C2_der_t, D0_der_t, D1_der_t, D2_der_t = self.params_der_t

        BL = (beta0_der_t + 2 * beta1_der_t * self.g_fun_phi(self.alpha_b1, i_str ** 0.5) + 2 * beta2_der_t * self.g_fun_phi(self.alpha_b1, i_str ** 0.5))
        CL = (C0_der_t + 2 * C1_der_t * self.g_fun_phi(self.alpha_c1, i_str) + 2 * C2_der_t * self.g_fun_phi(self.alpha_c2, i_str))
        DL = (D0_der_t + 2 * D1_der_t * self.g_fun_phi(self.alpha_d1, i_str ** 1.5) + 2 * D2_der_t * self.g_fun_phi(self.alpha_d2, i_str ** 1.5))

        a_h = wp.WaterPropertiesFineMillero(self.tk, self.pa).enthalpy_coefficient()

        val_1 = nu * z_prod * (a_h / (3 * self.b_param)) * np.log(1 + self.b_param * np.sqrt(i_str))
        val_2 = -2 * un.r_gas() * self.tk ** 2 * nu_prod * m * BL
        val_3 = -2 * un.r_gas() * self.tk ** 2 * nu_prod ** 1.5 * m ** 2 * CL
        val_4 = -2 * un.r_gas() * self.tk ** 2 * nu_prod ** 2 * m ** 3 * DL

        l_phi = val_1 + val_2 + val_3 + val_4

        return l_phi

    def heat_dilution(self, m1, m2):
        """
            Heat of dilution according to Silvester and Pitzer :cite:`Silvester1977`

            .. math::

                \\Delta \\bar{H}_{D}\\left(m_{1}\\rightarrow m_{2}\\right) = \\, ^\\phi L_{2} \\, -\\, ^\\phi L_{1}

            :return: heat of dilution in SI (float)
            """
        heat_dil = self.apparent_molal_enthalpy(m2) - self.apparent_molal_enthalpy(m1)

        return heat_dil
