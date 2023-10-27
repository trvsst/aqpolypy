"""
:module: ElectrolyteSolutionImplicitVirial
:platform: Unix, Windows, OS
:synopsis: Defines an electrolyte in Solvent with implicit solvent without pairing

.. moduleauthor::  Alex Travesset <trvsst@ameslab.gov>, Octobrt2023
.. history:
..
..
"""
import numpy as np
from scipy.special import xlogy as lg

from scipy.optimize import fsolve
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as fm
import aqpolypy.free_energy_polymer.ElectrolyteSolution as El


class ElectrolyteSolutionVirial(object):
    """
    Class defining an electrolyte solution described by the mean field model with implicit solvent
    """

    def __init__(self, ml, temp, b_param, e_s):

        """
        The constructor, with the following parameters

        :param ml: salt molality
        :param temp: temperature in Kelvin
        :param b_param: b-parameter for debye huckel contribution
        :param e_s: Virial coefficients
        """

        # molality
        self.ml = ml
        if not isinstance(self.ml, np.ndarray):
            raise ValueError('molality must be provided as a numpy array')

        # constants
        self.delta_w = un.delta_w()

        # temperature
        self.tp = temp

        # electrostatic parameter for debye-huckel
        self.b_param = b_param

        # virial coefficients
        self.e_s = e_s

        # water model at the given temperature and pressure
        self.press = 1.01325
        wfm = fm.WaterPropertiesFineMillero(self.tp, self.press)
        self.a_gamma = 3*wfm.a_phi()

        # square ionic strength
        self.sqrt_i_str = np.sqrt(self.ml)

        # these parameters are needed to instantiate ElectrolyteSolution but are not used
        v_r = 1 / (1e3*wfm.molar_volume())
        v_n = v_r
        v_w = 1 / un.mol_lit_2_mol_angstrom(v_r)
        v_s = 1 / un.mol_lit_2_mol_angstrom(v_n)
        de_w = 1800
        ds_w = 3.47
        mr_p, mr_m, mr_bp, mr_bm = [8.0, 8.0, 8.0, 8.0]
        param_w = {'v_w': v_w, 'de_w': de_w, 'ds_w': ds_w, 'de_2d': 0.0, 'ds_2d': 0.0, 'de_2a': 0.0, 'ds_2a': 0.0}
        dict_vol = {'v_s': v_s, 'v_b': v_s}
        dict_p = {'de_p0': de_w, 'ds_p0': 0.0, 'de_p1': de_w, 'ds_p1': 0.0, 'de_p2': de_w, 'ds_p2': 0.0}
        dict_bp = {'de_bp0': de_w, 'ds_bp0': 0.0, 'de_bp1': de_w, 'ds_bp1': 0.0, 'de_bp2': de_w, 'ds_bp2': 0.0}
        dict_m = {'de_m0': de_w, 'ds_m0': 0.0, 'de_m1': de_w, 'ds_m1': 0.0, 'de_m2': de_w, 'ds_m2': 0.0}
        dict_bm = {'de_bm0': de_w, 'ds_bm0': 0.0, 'de_bm1': de_w, 'ds_bm1': 0.0, 'de_bm2': de_w, 'ds_bm2': 0.0}
        dict_b = {'de_b': np.log(0.7), 'ds_b': 0.0}
        param_salt = {**dict_vol, **dict_p, **dict_bp, **dict_m, **dict_bm, **dict_b}
        dict_max = {'m_p': mr_p, 'm_m': mr_m, 'mb_p': mr_bp, 'mb_m': mr_bm}
        param_h = {**dict_max}

        self.el = El.ElectrolyteSolution(self.ml, temp, param_w, param_salt, param_h, b_param=b_param)

    def chem_water(self):
        """
        Water chemical potential
        """

        sol = np.zeros([16, self.ml.shape[0]])

        return -2*self.ml*self.osmotic_coeff()/self.delta_w

    def chem_salt_free(self):
        """
        Chemical Potential of free salt
        """

        return 2*np.log(self.ml)+2*self.log_gamma()

    def osmotic_coeff(self):
        """
        return osmotic coefficient
        """

        sol = np.zeros([16, self.ml.shape[0]])

        mu_1 = 1-0.5*self.delta_w*self.el.mu_w_debye(sol)/self.ml
        mu_2 = 0.5*self.e_s[0]*self.ml + 2*self.e_s[1]*self.ml**2/3.0

        return mu_1+mu_2

    def log_gamma(self):
        """
        return activity coefficient
        """

        sol = np.zeros([16, self.ml.shape[0]])
        mu_1 = 0.5*self.el.mu_sf_debye(sol)
        mu_2 = self.e_s[0]*self.ml + self.e_s[1]*self.ml** 2

        return mu_1+mu_2

    def gibbs_free(self):
        """
        Returns the Gibbs free energy normalized to the number of water molecules

        return Gibbs free energy
        """

        return self.chem_water()+self.ml*self.chem_salt_free()/self.delta_w
