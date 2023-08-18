"""
:module: ElectrolyteSolutionImplicitSolvent
:platform: Unix, Windows, OS
:synopsis: Defines an electrolyte in Solvent with implicit solvent

.. moduleauthor::  Alex Travesset <trvsst@ameslab.gov>, August2023
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

class ElectrolyteSolutionImplicit(object):
    """
    Class defining an electrolyte solution described by the mean field model with implicit solvent
    """

    def __init__(self, ml, temp, b_param, k_b, e_s):

        """
        The constructor, with the following parameters

        :param ml: salt molality
        :param temp: temperature in Kelvin
        :param b_param: b-parameter for debye huckel contribution
        :param k_b: Bjerrum constant
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

        # Bjerrum constant
        self.k_b = k_b

        # interaction parameters
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

        self.fract_b = None

    def chem_water(self):
        """
        Water chemical potential

        :param ml: molality
        :param f_B: fraction of Bjerrum pairs
        """

        sol = np.zeros([16, self.ml.shape[0]])
        sol[15] = self.fract_b
        mu_1 = self.el.mu_w_debye(sol) - (2-self.fract_b)*self.ml/un.delta_w()
        mu_2 = - 0.5*(self.e_s[0]*(1-self.fract_b)**2 + self.e_s[1]*self.fract_b**2)*self.ml**2/un.delta_w()

        return mu_1+mu_2

    def chem_salt_free(self):
        """
        Chemical Potential of free salt
        """

        sol = np.zeros([16, self.ml.shape[0]])
        sol[15] = self.fract_b

        mu = self.el.mu_sf_debye(sol) + 2*np.log((1-self.fract_b)*self.ml) + self.e_s[0] * (1-self.fract_b)*self.ml

        return mu

    def chem_salt_bjerrum(self):
        """
        Chemical Potential of Bjerrum salt
        """

        return chem_salt_bjerrum_f(self.ml, self.fract_b)

    def chem_salt_free_f(self, f_B):
        """
        Chemical Potential of free salt

        :param f_B: fraction of Bjerrum pairs
        """

        sol = np.zeros(16)
        sol[15] = f_B
        mu_salt_free = self.el.mu_sf_debye(sol) + 2*np.log((1-f_B)*self.ml) + self.e_s[0]*(1 - f_B)*self.ml

        return mu_salt_free

    def chem_salt_bjerrum_f(self, f_B):
        """
        Chemical Potential of Bjerrum salt

        :param f_B: fraction of Bjerrum pairs
        """

        mu_salt_bjerrum = np.log(f_B*self.ml) - np.log(self.k_b) + self.e_s[1]*f_B*self.ml

        return mu_salt_bjerrum

    def solve_bjerrum_equation(self):
        """
        Solves the Bjerrrum equation
        """

        f_b = np.zeros_like(self.ml)
        x0 = self.k_b*self.ml[0]

        def fun_bjerrum(y, *arg):
            ind = arg[0]
            val = self.chem_salt_free_f(y)[ind]-self.chem_salt_bjerrum_f(y)[ind]
            return val

        for ind, m_val in enumerate(self.ml):
            sol = fsolve(fun_bjerrum, x0, args=(ind,))
            x0 = sol[0]
            f_b[ind] = sol[0]

        self.fract_b = f_b

    def osmotic_coeff(self):
        """
        return osmotic coefficient
        """

        return -0.5*un.delta_w()*self.chem_water()/self.ml

    def log_gamma(self):
        """
        return activity coefficient
        """

        return 0.5*(self.chem_salt_free()-2*np.log(self.ml))
