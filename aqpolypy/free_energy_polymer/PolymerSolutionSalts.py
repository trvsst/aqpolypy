"""
:module: PolymerSolutionSalts
:platform: Unix, Windows, OS
:synopsis: Defines a Polymer in Solvent, including hydrogen bonds

.. moduleauthor:: Chi Yuanchen <ychi@iastate.edu>, July2020
.. history:
..                  Alex Travesset <trvsst@ameslab.gov>, July2020
..                  - changes
..
"""
import numpy as np
from scipy.special import xlogy as lg
from scipy.special import gamma as gamma

import aqpolypy.units.units as un
import aqpolypy.salts_theory.DebyeHuckel as dh
import aqpolypy.salts_theory.HardCore as hc


class PolymerSolutionSalts(object):
    """
    Defines a solution with polymers with hydrogen bonds generalizing the model :cite:`Dormidontova2002`
    """

    def __init__(self, v_p, v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac=np.array([1, 1])):
        """
        The constructor, with the following parameters

        :param v_p: polymer parameters :math:`(\\phi_p, \\frac{\\upsilon_w}{\\upsilon_p}, \\Delta F_p)`
        :param v_s: salt parameters (see definition below)
        :param v_w: volume of water in :math:`\\mbox{\\AA}^3`
        :param df_w: free energy change upon formation of hydrogen bond in water (in :math:`k_BT` units)
        :param x_ini: fraction of polymer hydrogen bonds
        :param p_ini: fraction of water hydrogen bonds
        :param n_k: number of Kuhn lengths for the polymer
        :param chi_p: Flory Huggins parameter
        :param chi_e: Virial parameter between polymer and electrolytes
        :param param_s: microscopic salt parameters :math:`(h_a, h_b, d_a, d_b)` (hydration number and diameter)
        :param b_o: object of water class  :class:`Bjerrum <aqpolypy.salts_theory.Bjerrum>`
        :param b_fac: B_factors for the different species in units of ion_size in hard-core repulsion

        the parameter v_s is given by
        :math:`(c_s, \\frac{\\upsilon_w}{\\upsilon_+}, \\frac{\\upsilon_w}{\\upsilon_-}, \\Delta F_a, \\Delta F_b)`

        where :math:`c_s` is the concentration in Molar units
        """

        # concentration in mols/litre
        self.conc = v_s[0]
        self.conc_ang = un.mol_lit_2_mol_angstrom(self.conc)

        # molecular volumes
        self.u_p = v_p[1]
        self.u_a = v_s[1]
        self.u_b = v_s[2]
        self.v_w = v_w

        # volume fractions
        self.phi_p = v_p[0]
        self.phi_a = self.conc_ang*self.v_w/self.u_a
        self.phi_b = self.conc_ang*self.v_w/self.u_b
        self.phi_w = 1 - self.phi_p - self.phi_a - self.phi_b
        self.phi_1 = self.phi_a + self.phi_b

        # polymer and polymer interaction parameters
        self.n = n_k
        self.chi_p = chi_p
        self.chi_e = chi_e

        # hydration numbers
        self.h_a = param_s[0]
        self.h_b = param_s[1]

        # fraction of hydration bonds (in water:x, in polymer:y)
        self.x = x_ini
        self.p = p_ini

        # fraction of water in each ion size
        self.f_a = self.h_a*self.u_a*self.phi_a/self.phi_w
        self.f_b = self.h_b*self.u_b*self.phi_b/self.phi_w

        # relative free energies of water association
        self.df_w = df_w
        self.df_p = v_p[2]
        self.df_a = v_s[3]
        self.df_b = v_s[4]

        self.i_size_a = param_s[2]
        self.i_size_b = param_s[3]
        self.i_size = 0.5*(self.i_size_a+self.i_size_b)

        # salt free energy
        self.bjerrum_object = b_o
        self.dh_free = dh.DebyeHuckel(self.bjerrum_object)

        # hard core free energy
        self.b_fac = b_fac
        self.hc_free = hc.HardCore(b_fac=self.b_fac)

        # Derivative of parameters
        self.df_adp = self.f_a / self.phi_w
        self.df_ada = self.f_a * (1 / self.phi_a + 1 / self.phi_w)
        self.df_adb = self.f_a / self.phi_w

        self.df_bdp = self.f_b / self.phi_w
        self.df_bda = self.f_b / self.phi_w
        self.df_bdb = self.f_b * (1 / self.phi_b + 1 / self.phi_w)

        self.dxdp = - self.x / self.phi_p
        self.dxda = 0
        self.dxdb = 0

        self.dydp = self.p / self.phi_w
        self.dyda = self.p / self.phi_w
        self.dydb = self.p / self.phi_w

    def free(self):
        """
        Free energy of a polymer in solution :math:`\\frac{F \\upsilon}{V k_B T}=\\frac{f}{k_B T}`
        (per unit volume (volume in units of :math:`\\upsilon_w` and in :math:`k_B T` units)

        :return: value of free energy (float)
        """

        f_ref_11 = lg(self.u_p * self.phi_p / self.n, self.phi_p / (self.n * np.exp(1)))
        f_ref_12 = lg(self.u_a * self.phi_a, self.phi_a / np.exp(1))
        f_ref_13 = lg(self.u_b * self.phi_b, self.phi_b / np.exp(1))
        f_ref_2 = lg(self.phi_w, self.phi_w / np.exp(1))

        f_int_1 = self.chi_p * self.phi_p * self.phi_w
        f_int_2 = self.chi_e * self.phi_p * self.phi_1
        f_as_1_1 = (lg(self.x, self.x) + lg(1 - self.x, 1 - self.x)
                    - self.x * self.df_p)
        f_as_1 = 2 * self.u_p * self.phi_p * f_as_1_1

        f_as_2_1 = lg(self.p, self.p) + lg(1 - self.f_a - self.p, 1 - self.f_a - self.p) - self.p * self.df_w
        f_as_2 = 2 * self.phi_w * f_as_2_1

        z_val = (1 - self.f_b - self. p - self.x * self.u_p * self.phi_p / self.phi_w)
        f_as_3 = 2 * self.phi_w * lg(z_val, z_val)
        f_as_4 = -2 * self.phi_w * lg(self.x * self.u_p * self.phi_p / self.phi_w + self.p, 2 * self.phi_w / np.exp(1))
        f_as_5_1 = (np.log(gamma(self.h_a + 1)) - self.df_a) * self.f_a
        f_as_5_2 = (np.log(gamma(self.h_b + 1)) - self.df_b) * self.f_b
        f_as_5 = self.phi_w * (f_as_5_1 + f_as_5_2)
        f_as_6_1 = lg(1 - self.f_a - self.f_b, 1 - self.f_a - self.f_b)
        f_as_6_2 = lg(1 - self.f_a, 1 - self.f_a) + lg(1 - self.f_b, 1 - self.f_b)
        f_as_6 = self.phi_w * (f_as_6_1 - f_as_6_2)
        f_as_7 = self.phi_w * (self.f_a + self.f_b) * np.log(self.phi_w / np.exp(1))

        f_dh = self.dh_free.free_energy_db_excess(self.conc, self.i_size)
        f_hc = np.sum(self.hc_free.free_energy_hc_excess(self.conc, self.i_size))

        f_ref_all = f_ref_11 + f_ref_12 + f_ref_13 + f_ref_2
        f_int_all = f_int_1 + f_int_2
        f_as_all = f_as_1 + f_as_2 + f_as_3 + f_as_4 + f_as_5 + f_as_6 + f_as_7

        return f_ref_all + f_int_all + f_as_all + f_dh + f_hc

    def chem_potential_w(self):
        """
        Reduced chemical potential  for water :math:`\\frac{\\mu_w}{k_B T}=\\frac{1}{k_B T}(f- \\phi_p \\frac{\\partial f}{\\partial \\phi_p}- \\phi_+ \\frac{\\partial f}{\\partial \\phi_+}- \\phi_- \\frac{\\partial f}{\\partial \\phi_-}`

        :return: value of chemical potential (float)
        """

        mu_0 = - self.u_p * self.phi_p / self.n - self.u_a * self.phi_a - self.u_b * self.phi_b    
        mu_1_1 = np.log(self.phi_w) - self.phi_w
        mu_1_2 = self.chi_p * self.phi_p * (self.phi_p + self.phi_1) - self.chi_e * self.phi_1 * self.phi_p

        mu_2 = 0
        mu_3 = 2 * (lg(1 - self.f_a - self.p, 1 - self.f_a - self.p)
               + lg(self.p, self.p) - self.p * self.df_w)

        z_val = (1 - self.f_b - self. p - self.x * self.u_p * self.phi_p / self.phi_w)
        #mu_4 = 2 * lg(z_val, z_val)
        #mu_5 = 2 * (self.phi_p + self.phi_1) * self.x * self.u_p *\
        #       (1 - self.phi_1) / self.phi_w * np.log(z_val)
        mu_4 = 2 * (1 - self.f_b - self.p) * np.log(z_val) - 2 * self.p * np.log(2 * self.phi_w / np.exp(1)) -\
               2 * self.p * (1 - self.phi_w) + 2 * self.x * self.u_p * self.phi_p
        mu_5 = 0

        mu_6 = (np.log(gamma(self.h_a + 1)) - self.df_a) * self.f_a +\
               (np.log(gamma(self.h_b + 1)) - self.df_b) * self.f_b
        mu_7 = (lg(1 - self.f_a - self.f_b, 1 - self.f_a - self.f_b) -\
               lg(1 - self.f_a, 1 - self.f_a) - lg(1 - self.f_b, 1 - self.f_b))

        mu_8 = - (self.f_a + self.f_b) * (np.log(self.phi_w) - self.phi_w)
        #mu_9 = - 2 * (- self.x * self.u_p * self.phi_1 + self.p) * np.log(2 * self.phi_w) +\
        #       2 * self.phi_w * (self.x * self.u_p * self.phi_p / self.phi_w + self.p)
        mu_9 = 0

        mu_10_a = - 2 * self.u_p * self.phi_p 
        mu_10_b = (self.phi_p * self.dxdp + self.phi_a * self.dxda + self.phi_b * self.dxdb) 
        mu_10_c = np.log(self.x / (1 - self.x) * np.exp(- self.df_p) / z_val / 2 / self.phi_w)
        mu_10 = mu_10_a * mu_10_b * mu_10_c

        mu_11_a = - 2 * self.phi_w
        mu_11_b = (self.phi_p * self.dydp + self.phi_a * self.dyda + self.phi_b * self.dydb)
        mu_11_c = np.log(self.p / (1 - self.f_a - self.p) * np.exp(- self.df_w) / z_val / 2 / self.phi_w)
        mu_11 = mu_11_a * mu_11_b * mu_11_c

        mu_12_a = self.phi_p * self.df_adp + self.phi_a * self.df_ada + self.phi_b * self.df_adb
        mu_12_b_1 = (1 - self.f_a) / (1 - self.f_a - self.f_b)
        mu_12_b_2 = np.exp(- self.df_a - 1 + np.log(gamma(self.h_a + 1)))
        mu_12_b_3 = (1 - self.f_a - self.p)**2 * self.phi_w
        mu_12_b = np.log(mu_12_b_1 * mu_12_b_2 / mu_12_b_3)
        mu_12 = - self.phi_w * mu_12_a * mu_12_b

        mu_13_a = self.phi_p * self.df_bdp + self.phi_a * self.df_bda + self.phi_b * self.df_bdb
        mu_13_b = (1 - self.f_b) / (1 - self.f_a - self.f_b) 
        mu_13_c = np.exp(- self.df_b - 1 + np.log(gamma(self.h_b + 1)))
        mu_13_d = z_val ** 2 * self.phi_w
        mu_13 = - self.phi_w * mu_13_a *np.log(mu_13_b * mu_13_c / mu_13_d)

        mu_dh = self.dh_free.pot_chem_db_excess(self.conc, self.i_size)
        mu_hc = np.sum(self.hc_free.pot_chem_hc_excess(self.conc, self.i_size))

        return mu_0 + mu_1_1 + mu_1_2 + mu_2 + mu_3 + mu_4 + mu_5 + mu_6 + mu_7 + mu_8 + mu_9 + mu_10 + mu_11 + mu_12 + mu_13 + mu_dh + mu_hc

    def chem_potential_pm(self):
        """
        Reduced chemical potential  for the ions :math:`\\frac{1}{2}( \\frac{ \\upsilon_+}{ \\upsilon_w}  \\frac{ \\mu_w}{k_B T} +  \\frac{ \\upsilon_+}{ \\upsilon_w} \\frac{1}{k_B T} \\frac{ \\partial f}{ \\partial  \\phi_+}+ \\frac{ \\upsilon_-}{ \\upsilon_w}  \\frac{ \\mu_w}{k_B T} +  \\frac{ \\upsilon_-}{ \\upsilon_w} \\frac{1}{k_B T} \\frac{ \\partial f}{ \\partial  \\phi_-})`

        :return: value of chemical potential (float)
        """
        u_pm = (1 / self.u_a + 1 / self.u_b)
        z_val = (1 - self.f_b - self.p - self.x * self.u_p * self.phi_p / self.phi_w) 

        mu_0 = - u_pm * self.u_p / 2 * self.phi_p / self.n - u_pm * self.u_a / 2 * self.phi_a - u_pm * self.u_b / 2 * self.phi_b
        mu_1_1 = np.log(self.phi_a * self.phi_b) / 2 - u_pm / 2 * self.phi_w
        #mu_1_2 = - u_pm / 2 * self.chi_p * self.phi_w * self.phi_p - u_pm / 2 * self.chi_e * self.phi_p * (1 - self.phi_p)
        mu_1_2 = u_pm * self.phi_p / 2 * (- self.chi_p * self.phi_w + self.chi_e * (1 - self.phi_1 ))

        #mu_2 = - self.x * u_pm * self.u_p * (1 - self.phi_1) * np.log(z_val)
        mu_2 =  u_pm * (self.p * self.phi_w + self.x * self.u_p * self.phi_p)

        #mu_3 = - self.x * u_pm * self.u_p * (1 - self.phi_1) * np.log(2 * self.phi_w) + u_pm * self.phi_w * (self.x * self.u_p *self.phi_p / self.phi_w - self.p)
        mu_3 = u_pm / 2 *(self.f_a + self.f_b) * self.phi_w

        mu_4 = - self.u_p * self.phi_p * (self.dxdp * u_pm * self.phi_p + self.dxda * (u_pm * self.phi_a - 1/ self.u_a) + self.dxdb * (u_pm * self.phi_b - 1/ self.u_b)) * \
               np.log(self.x / (1 - self.x) * np.exp(- self.df_p) / z_val / 2 / self.phi_w)

        mu_5 = - self.phi_w * (self.dydp * u_pm * self.phi_p + self.dyda * (u_pm * self.phi_a - 1/ self.u_a) + self.dydb * (u_pm * self.phi_b - 1/ self.u_b)) * \
               np.log(self.p / (1 - self.f_a - self.p) * np.exp(- self.df_w) / z_val / 2 / self.phi_w)

        mu_6 = - self.phi_w / 2 * (self.df_adp * u_pm * self.phi_p + self.df_ada * (u_pm * self.phi_a - 1/ self.u_a) + self.df_adb * (u_pm * self.phi_b - 1/ self.u_b)) * \
               np.log((1 - self.f_a) / (1 - self.f_a - self.f_b) * np.exp(- self.df_a - 1 + np.log(gamma(self.h_a + 1))) / (1 - self.f_a - self.p) ** 2 / self.phi_w)

        mu_7 = - self.phi_w / 2 * (self.df_bdp * u_pm * self.phi_p + self.df_bda * (u_pm * self.phi_a - 1/ self.u_a) + self.df_bdb * (u_pm * self.phi_b - 1/ self.u_b)) * \
               np.log((1 - self.f_b) / (1 - self.f_a - self.f_b) *  np.exp(- self.df_b - 1 + np.log(gamma(self.h_b + 1))) / z_val ** 2 / self.phi_w)


        mu_dh = self.dh_free.pot_chem_db_excess(self.conc, self.i_size)
        mu_hc = self.hc_free.pressure_hc_excess(self.conc, self.i_size)
        
        mu_hc_0 = 0.5 * (- mu_hc[0] / self.phi_a / self.u_a - mu_hc[1] / self.phi_b / self.u_b + u_pm * np.sum(mu_hc))
        
        mu_dh_0 = 0.5 * (- mu_dh / (self.conc_ang*self.v_w) + u_pm * mu_dh)
        
        
        

        return mu_0 + mu_1_1 + mu_1_2 + mu_2 + mu_3 + mu_4 + mu_5 + mu_6 + mu_7 + mu_dh_0 + mu_hc_0

    def chem_potential_p(self):
        """
        Reduced chemical potential  for the polymers :math:` \\frac{N \\upsilon_p}{ \\upsilon_w}  \\frac{ \\mu_w}{k_B T} +  \\frac{N \\upsilon_p}{ \\upsilon_w} \\frac{1}{k_B T} \\frac{ \\partial f}{ \\partial  \\phi_p}`

        :return: value of chemical potential (float)
        """

        z_val = (1 - self.f_b - self.p - self.x * self.u_p * self.phi_p / self.phi_w)
        #mu_1_1 = np.log(self.phi_p / self.n / np.exp(1)) - self.n * self.u_a / self.u_p * self.phi_a - self.n * self.u_b / self.u_p * self.phi_b
        
        mu_1_1 =  np.log(self.phi_p / self.n) - self.n / self.u_p * (self.phi_w + self.u_a * self.phi_a + self.u_b * self.phi_b) - self.phi_p
        mu_1_2 =  self.n / self.u_p * (1 - self.phi_p) * (self.chi_p * self.phi_w  + self.chi_e * self.phi_1)

        mu_2 = 2 * self.n * (lg(1 - self.x, 1 - self.x) + lg(self.x, self.x) - self.x * self.df_p)
        #mu_3 = - 2 * self.n * self.x * (1 - self.phi_1) * np.log(z_val)
        mu_3 = - 2 * self.n * self.x * (np.log(z_val) + 1)

        mu_4 = self.n / self.u_p * (self.f_a + self.f_b) * self.phi_w

        #mu_5 = - 2 * self.n * self.x * (1 - self.phi_1) * np.log(2 * self.phi_w) + 2 * self.n / self.u_p * (self.x * self.u_p - self.p) * self.phi_w
        mu_5 = -2 * self.n / self.u_p * (self.x * self.u_p * (np.log(2 * self.phi_w / np.exp(1)) - self.phi_p) - self.p * self.phi_w)

        mu_6 = - 2 * self.n * self.phi_p * ((self.phi_p-1) * self.dxdp + self.phi_a * self.dxda + self.phi_b * self.dxdb) * \
               np.log(self.x / (1 - self.x) * np.exp(- self.df_p) / z_val / 2 / self.phi_w)

        mu_7 = - 2 * self.n / self.u_p * self.phi_w * ((self.phi_p-1) * self.dydp + self.phi_a * self.dyda + self.phi_b * self.dydb) * \
               np.log(self.p / (1 - self.f_a - self.p) * np.exp(- self.df_w) / z_val / 2 / self.phi_w)

        mu_8 = - self.n / self.u_p * self.phi_w * ((self.phi_p-1) * self.df_adp + self.phi_a * self.df_ada + self.phi_b * self.df_adb) * \
               np.log((1 - self.f_a) / (1 - self.f_a - self.f_b) * np.exp(- self.df_a) / (1 - self.f_a - self.p) ** 2 / self.phi_w)

        mu_9 = - self.n / self.u_p * self.phi_w * ((self.phi_p-1) * self.df_bdp + self.phi_a * self.df_bda + self.phi_b * self.df_bdb) * \
               np.log((1 - self.f_b) / (1 - self.f_a - self.f_b) * np.exp(- self.df_b) / z_val ** 2 / self.phi_w)

        mu_dh = self.dh_free.pot_chem_db_excess(self.conc, self.i_size)
        mu_hc = np.sum(self.hc_free.pot_chem_hc_excess(self.conc, self.i_size))
        
        mu_10 = self.n / self.p * (mu_dh + mu_hc)

        return mu_1_1 + mu_1_2 + mu_2 + mu_3 + mu_4 + mu_5 + mu_6 + mu_7 + mu_8 + mu_9 + mu_10

