"""
:module: SaltPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as wp
from abc import ABC, abstractmethod


class SaltProperties(ABC):
    def __init__(self, m, tk, pa=1):
        """
        constructor

        :param m: molality
        :param tk: temperature absolute
        :param pa: pressure in atm
        :instantiate:

        """

        # temperature and pressure
        self.tk = tk
        self.pa = pa
        self.m = m

        # Calculations
        """
        Calculations for apparent molal volume (short hand parametric form)
        """
        # units are (kg/mol)^{1/2}
        self.b_param = 1.2
        self.hf = 0.5 * np.log(1 + self.b_param * np.sqrt(self.i_str)) / self.b_param

        """
        Calculations h_fun_gamma
        """
        self.hfg = 4 * self.hf(self.i_str) + np.sqrt(self.i_str) / (1 + self.b_param * np.sqrt(self.i_str))

        """
        Calculations p_fun_gamma
        """
        # units are (kg/mol)^{1/2}
        self.alpha = 2.0

        self.x = self.alpha * np.sqrt(self.i_str)

        self.pfg = (1 - (1 + self.x - 0.5 * self.x ** 2) * np.exp(-self.x)) / self.x ** 2

        """
        Calculations params
        """
        self.pr = self.pa * un.atm_2_bar(1)
        self.pr_atm = un.atm_2_bar(1)

        self.vp_0 = self.cm[0] + self.cm[1] * self.tk + self.cm[2] * self.tk ** 2 + self.cm[3] * self.tk ** 3
        self.vp_1 = (self.pr - self.pr_atm) * (self.cm[4] + self.cm[5] * self.tk + self.cm[6] * self.tk ** 2)
        self.vp_2 = (self.pr - self.pr_atm) ** 2 * (self.cm[7] + self.cm[8] * self.tk)

        self.vp = self.vp_0 + self.vp_1 + self.vp_2

        self.bp_0 = self.cm[9] + self.cm[10] / (self.tk - 227) + self.cm[11] * self.tk + self.cm[12] * self.tk ** 2 + \
                    self.cm[13] / (680 - self.tk)
        self.bp_1_1 = self.cm[14] + self.cm[15] / (self.tk - 227) + self.cm[16] * self.tk + self.cm[17] * self.tk ** 2
        self.bp_1 = (self.bp_1_1 + self.cm[18] / (680 - self.tk)) * (self.pr - self.pr_atm)
        self.bp_2_1 = self.cm[19] + self.cm[20] / (self.tk - 227) + self.cm[21] * self.tk + self.cm[22] / (
                    680 - self.tk)
        self.bp_2 = self.bp_2_1 * (self.pr - self.pr_atm) ** 2

        self.bp = self.bp_0 + self.bp_1 + self.bp_2

        self.cq = self.cm[23] + self.cm[24] / (self.tk - 227) + self.cm[25] * self.tk + self.cm[26] * self.tk ** 2 + \
                  self.cm[27] / (680 - self.tk)
        self.cp = 0.5 * self.cq
        self.param = np.array([self.vp, self.bp, self.cp])

        """
        Calculations for stoichiometry coefficients
        """
        # nu_+ + nu_-
        self.nu = np.sum(self.mat_stoich[0])
        # nu_+ nu_-
        self.nu_prod = self.mat_stoich[0, 0] * self.mat_stoich[0, 1]
        # abs(z_+ z_-)
        self.z_prod = np.abs(self.mat_stoich[1, 0] * self.mat_stoich[1, 1])
        # nu_+, z_+
        self.nz_prod_plus = self.mat_stoich[0, 0] * self.mat_stoich[1, 0]

        self.mat = np.array([self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus])

        """
        Calculation for Ionic strength
        """
        self.i_str = 0.5 * self.m * np.sum(self.mat_stoich[0] * self.mat_stoich[1] ** 2)

        """
        Calculations for molar volume (infinite dilution)
        """
        # the factor 10 is the conversion from J to bar cm^3
        self.ct = 10 * un.r_gas() * self.tk

        # molar volume water in cm^3/mol
        self.vol_water = 1e6 * wp.WaterPropertiesFineMillero(self.tk, self.pa).molar_volume()

        # stoichiometric_coefficients
        self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus = self.mat
        self.m_r = self.p_ref[1]
        self.y_r = self.p_ref[2]

        self.vp, self.bp, self.cp = self.param

        self.mv_i_0 = self.vp / self.m_r - self.y_r * self.vol_water
        self.mv_i_1 = -self.nu * self.z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.hf
        self.mv_i_2 = -2 * self.nu_prod * self.ct * (self.m_r * self.bp + self.nz_prod_plus * self.m_r ** 2 * self.cp)
        # this is in cm^3/mol
        self.mv_i = self.mv_i_0 + self.mv_i_1 + self.mv_i_2

        # return in SI m^3
        self.mol_vol_inf_dil = 1e-6 * self.mv_i

        """
        Calculations density_sol
        """
        self.mw = wp.WaterPropertiesFineMillero(self.tk, self.pa).MolecularWeight
        # convert to cm^3/mol
        self.v_water = 1e6 * wp.WaterPropertiesFineMillero(self.tk, self.pa).molar_volume()

        # convert to cm^3/mol
        self.v_salt = 1e6 * self.mol_vol
        self.mw_salt = self.p_ref[0]

        # density in g/cm^3
        self.dens = self.mw * (1 + 1e-3 * self.m * self.mw_salt) / (
                self.v_water + 1e-3 * self.m * self.mw * self.v_salt)

        # return density in kg/m3
        self.dens_sol = 1e3 * self.dens

        """
        Calculations molar_vol
        """
        # the factor 10 is the conversion from J to bar cm^3
        self.ct = 10 * un.r_gas() * self.tk

        # stoichiometric_coefficients
        self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus = self.mat

        # coefficients V_m, B, C
        self.vp, self.bp, self.cp = self.param

        # infinite molar volume, convert to cm^3/mol
        self.v_1 = 1e6 * self.mol_vol_inf_dil

        self.val_1 = self.v_1 + self.nu * self.z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.hf
        self.val_2 = 2 * self.nu_prod * self.ct * (self.bp * self.m + self.nz_prod_plus * self.cp * self.m ** 2)

        # molar volume in cm^3/mol
        self.val = self.val_1 + self.val_2

        # return in m^3/mol
        self.mol_vol = 1e-6 * self.val

        """
        Calculations osmotic_coeff
        """
        # pressure is 1 atm
        self.press = 1
        self.tc = 298.15

        self.beta0_1 = self.cq[0] + self.cq[1] * (1 / self.tk - 1 / self.tc) + self.cq[2] * np.log(self.tk / self.tc)
        self.beta0_2 = self.cq[3] * (self.tk - self.tc) + self.cq[4] * (self.tk ** 2 - self.tc ** 2)
        self.beta0 = self.beta0_1 + self.beta0_2

        self.beta1 = self.cq[5] + self.cq[8] * (self.tk - self.tc) + self.cq[9] * (self.tk ** 2 - self.tc ** 2)

        self.c_phi_1 = self.cq[10] + self.cq[11] * (1 / self.tk - 1 / self.tc) + self.cq[12] * np.log(self.tk / self.tc)
        self.c_phi = self.c_phi_1 + self.cq[13] * (self.tk - self.tc)

        # stoichiometric_coefficients
        self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus = self.mat

        self.x = np.sqrt(self.i_str)

        self.val_1 = -self.z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_phi() * self.x / (
                1 + 1.2 * self.x)
        self.val_2 = 2 * self.m * (self.nu_prod / self.nu) * (self.beta0 + self.beta1 * np.exp(-2 * self.x))
        self.val_3 = (2 * self.nu_prod ** 1.5 / self.nu) * self.m ** 2 * self.c_phi

        self.osmotic_coefficient = 1 + self.val_1 + self.val_2 + self.val_3

        """
        Calculations log_gamma
        """
        # pressure is 1 atm
        self.press = 1
        self.tc = 298.15

        self.beta0_1 = self.cq[0] + self.cq[1] * (1 / self.tk - 1 / self.tc) + self.cq[2] * np.log(self.tk / self.tc)
        self.beta0_2 = self.cq[3] * (self.tk - self.tc) + self.cq[4] * (self.tk ** 2 - self.tc ** 2)
        self.beta0 = self.beta0_1 + self.beta0_2

        self.beta1 = self.cq[5] + self.cq[8] * (self.tk - self.tc) + self.cq[9] * (self.tk ** 2 - self.tc ** 2)

        self.c_phi_1 = self.cq[10] + self.cq[11] * (1 / self.tk - 1 / self.tc) + self.cq[12] * np.log(self.tk / self.tc)
        self.c_phi = self.c_phi_1 + self.cq[13] * (self.tk - self.tc)

        # stoichiometric_coefficients
        self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus = self.mat

        self.val_1 = -self.z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_phi() * self.hfg
        self.val_2 = 2 * self.m * (self.nu_prod / self.nu) * (2 * self.beta0 + 2 * self.beta1 * self.pfg)
        self.val_3 = 1.5 * (2 * self.nu_prod ** 1.5 / self.nu) * self.m ** 2 * self.c_phi

        self.log_g = self.val_1 + self.val_2 + self.val_3

        """
        Calculations log_gamma_simple
        """
        self.a_limiting = 0.5108
        self.b_star = self.cf[0]
        self.beta = self.cf[1]
        self.c = self.cf[2]
        self.d = self.cf[3]

        # this is the value in base log_{10}
        self.val = -self.a_limiting * np.sqrt(self.x) / (1 + self.b_star * np.sqrt(
            self.x)) + self.beta * self.x + self.c * self.x ** 2 + self.d * self.x ** 3

        # return the value in ln
        self.lgs = self.val * np.log(10)

    @abstractmethod
    def h_fun(self):
        """
        Abstract method:
        """
        pass

    @abstractmethod
    def h_fun_gamma(self):
        """
        Abstract method:
        """
        pass

    @abstractmethod
    def p_fun_gamma(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def params(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def stoichiometry_coeffs(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def ionic_strength(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def molar_vol_infinite_dilution(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def density_sol(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def molar_vol(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def osmotic_coeff(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def log_gamma(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def log_gamma_simple(self):
        """
        Abstract method:
        """

        pass

    @abstractmethod
    def log_gamma_nacl_simple(self):
        """
        Abstract method:
        """
        pass

    @abstractmethod
    def nacl_params(self):
        """
        Abstract method:
        """
        pass
