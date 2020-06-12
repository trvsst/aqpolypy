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
    def __init__(self, tk, pa=1):
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

        # Calculations for stoichiometry coefficients
        self.mat_stoich = np.array([[None, None], [None, None]])
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
    def ionic_strength(self, m):
        """
        Abstract method:
        """
        i_str = 0.5 * m * np.sum(self.mat_stoich[0] * self.mat_stoich[1] ** 2)
        return i_str

    @abstractmethod
    def molar_vol_infinite_dilution(self):
        """
        Abstract method:
        """
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # molar volume water in cm^3/mol
        vol_water = 1e6 * wp.WaterPropertiesFineMillero(self.tk, self.pa).molar_volume()

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat
        m_r = self.p_ref[1]
        y_r = self.p_ref[2]

        vp, bp, cp = self.param

        mv_i_0 = vp / m_r - y_r * vol_water
        mv_i_1 = -nu * z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.hf
        mv_i_2 = -2 * self.nu_prod * self.ct * (m_r * self.bp + self.nz_prod_plus * m_r ** 2 * cp)
        # this is in cm^3/mol
        mv_i = mv_i_0 + mv_i_1 + mv_i_2

        # return in SI m^3
        mol_vol_inf_dil = 1e-6 * mv_i
        return mol_vol_inf_dil

    @abstractmethod
    def density_sol(self, m):
        """
        Abstract method:
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
        dens_sol = 1e3 * dens
        return dens_sol

    @abstractmethod
    def molar_vol(self, m):
        """
        Abstract method:
        """
        # the factor 10 is the conversion from J to bar cm^3
        ct = 10 * un.r_gas() * self.tk

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        # coefficients V_m, B, C
        vp, bp, cp = self.param

        # infinite molar volume, convert to cm^3/mol
        v_1 = 1e6 * self.molar_vol_infinite_dilution()

        val_1 = v_1 + nu * z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_v() * self.hf
        val_2 = 2 * nu_prod * ct * (bp * m + nz_prod_plus * cp * m ** 2)

        # molar volume in cm^3/mol
        val = val_1 + val_2

        # return in m^3/mol
        mol_vol = 1e-6 * val
        return mol_vol

    @abstractmethod
    def osmotic_coeff(self, m):
        """
        Abstract method:
        """
        # pressure is 1 atm
        press = 1
        tc = 298.15

        beta0_1 = self.cq[0] + self.cq[1] * (1 / self.tk - 1 / tc) + self.cq[2] * np.log(self.tk / tc)
        beta0_2 = self.cq[3] * (self.tk - tc) + self.cq[4] * (self.tk ** 2 - tc ** 2)
        beta0 = beta0_1 + beta0_2

        beta1 = self.cq[5] + self.cq[8] * (self.tk - tc) + self.cq[9] * (self.tk ** 2 - tc ** 2)

        c_phi_1 = self.cq[10] + self.cq[11] * (1 / self.tk - 1 / tc) + self.cq[12] * np.log(self.tk / tc)
        c_phi = c_phi_1 + self.cq[13] * (self.tk - tc)

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        x = np.sqrt(self.i_str(m))

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_phi() * x / (1 + 1.2 * x)
        val_2 = 2 * m * (nu_prod / nu) * (beta0 + beta1 * np.exp(-2 * x))
        val_3 = (2 * nu_prod ** 1.5 / nu) * m ** 2 * c_phi

        osmotic_coefficient = 1 + val_1 + val_2 + val_3
        return osmotic_coefficient

    @abstractmethod
    def log_gamma(self, m):
        """
        Abstract method:
        """
        # pressure is 1 atm
        press = 1
        tc = 298.15

        beta0_1 = self.cq[0] + self.cq[1] * (1 / self.tk - 1 / tc) + self.cq[2] * np.log(self.tk / tc)
        beta0_2 = self.cq[3] * (self.tk - tc) + self.cq[4] * (self.tk ** 2 - tc ** 2)
        beta0 = beta0_1 + beta0_2

        beta1 = self.cq[5] + self.cq[8] * (self.tk - tc) + self.cq[9] * (self.tk ** 2 - tc ** 2)

        c_phi_1 = self.cq[10] + self.cq[11] * (1 / self.tk - 1 / tc) + self.cq[12] * np.log(self.tk / tc)
        c_phi = c_phi_1 + self.cq[13] * (self.tk - tc)

        # stoichiometric_coefficients
        nu, nu_prod, z_prod, nz_prod_plus = self.mat

        val_1 = -z_prod * wp.WaterPropertiesFineMillero(self.tk, self.pa).a_phi() * self.hfg
        val_2 = 2 * m * (nu_prod / nu) * (2 * beta0 + 2 * beta1 * self.pfg)
        val_3 = 1.5 * (2 * nu_prod ** 1.5 / nu) * m ** 2 * c_phi

        log_g = val_1 + val_2 + val_3
        return log_g
