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
    Defines a solution with polymers with hydrogen bonds following the model :cite:`Dormidontova2002`
    """

    def __init__(self, v_p, v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac=np.array([1, 1])):
        """
        The constructor

        we use the objects:
        v_p :math:`(\\phi_p, \\frac{\\upsilon_w}{\\upsilon_p}, \\Delta F_p)`
        v_s :math:`(c_s, \\frac{\\upsilon_w}{\\upsilon_+}, \\frac{\\upsilon_w}{\\upsilon_-}, \\Delta F_a, \\Delta F_p)`

        here c_s is salt concentration in M units

        also

        param_s :math:`(h_a, h_b, size_a, size_b)`

        :param v_p: polymer parameters
        :param v_s: salt parameters
        :param v_w: volume of water in :math:`a^3`
        :param df_w: free energy change upon formation of hydrogen bond in water (in :math:`k_BT` units)
        :param x_ini: fraction of polymer hydrogen bonds
        :param p_ini: fraction of water hydrogen bonds
        :param n_k: number of Kuhn lengths for the polymer
        :param chi_p: a Flory Huggins parameter with the temperature dependence :math:`\\chi_p=A_p+\\frac{B_p}{T}`
        :param chi_e: a Flory Huggins parameter between polymer and electrolytes
        :param param_s: microscopic salt parameters
        :param b_o: object of water class  :class:`Bjerrum <aqpolypy.salts_theory.Bjerrum>`
        :param b_fac: B_factors for the different species in units of ion_size to calculate the freen energy of hard-core repulsion 
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
        self.dh_free = dh.DebyeHuckel(self.bjerrum_object).free_energy_db_excess(self.conc, self.i_size)

        # hard core free energy
        self.b_fac = b_fac
        self.hc_free = hc.HardCore(b_fac=self.b_fac).free_energy_hc_excess(self.conc, self.i_size)

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
        f_as_1 = 2 * self.u_p * self.phi_p * (lg(self.x, self.x) + lg(1 - self.x, 1 - self.x) - self.x * self.df_p)
        f_as_2 = 2 * self.phi_w * (lg(self.p, self.p) + lg(1 - self.f_a - self.p, 1 - self.f_a - self.p) - self.p * self.df_w)

        z_val = (1 - self.f_b - self. p - self.x * self.u_p * self.phi_p) / self.phi_w
        f_as_3 = 2 * self.phi_w * lg(z_val, z_val)
        f_as_4 = -2 * self.phi_w * lg(self.x * self.u_p * self.phi_p / self.phi_w + self.p, 2 * self.phi_w / np.exp(1))
        f_as_5 = self.phi_w * ((np.log(gamma(self.h_a + 1)) - self.df_a) * self.f_a + (np.log(gamma(self.h_b + 1)) - self.df_b) * self.f_b)
        f_as_6 = self.phi_w * (lg(1 - self.f_a - self.f_b, 1 - self.f_a - self.f_b) + lg(1 - self.f_a, 1 - self.f_a) + lg(1 - self.f_b, 1 - self.f_b))
        f_as_7 = self.phi_w * (self.f_a + self.f_b) * np.log(self.phi_w / np.exp(1))

        f_dh = self.dh_free
        f_hc = self.hc_free

        return f_ref_11 + f_ref_12 + f_ref_13 + f_ref_2 + f_int_1 + f_int_2 + f_as_1 + f_as_2 + f_as_3 + f_as_4 + f_as_5 + f_as_6 + f_as_7 + f_dh + f_hc
