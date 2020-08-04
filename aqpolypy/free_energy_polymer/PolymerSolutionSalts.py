"""
:module: PolymerSolution
:platform: Unix, Windows, OS
:synopsis: Defines a Polymer in Solvent, including hydrogen bonds

.. moduleauthor:: CHI YUANCHEN <ychi@iastate.edu>, July2020
.. history:
..
"""
import numpy as np
from scipy.special import xlogy as lg
from scipy.special import gamma as gamma

import aqpolypy.salts_theory.DebyeHuckel as dh
import aqpolypy.salts_theory.HardCore as hc


class PolymerSolutionSalts(object):
    """
    Defines a solution with polymers with hydrogen bonds following the model :cite:`Dormidontova2002`
    """

    def __init__(self, phi_p, phi_a, phi_b, x, p, n_kuhn, u_p, u_a, u_b, f_a, f_b, chi_p, chi_e, df_w, df_p, df_a, df_b, h_a, h_b, bjerrum_object, b_fac=np.array([1, 1, 0])):
        """
        The constructor

        :param phi_p: polymer fraction :math:`\\phi_p`
        :param phi_a: salt fraction :math:`\\phi_+`
        :param phi_b: salt fraction :math:`\\phi_-`
        :param x_ini: fraction of polymer hydrogen bonds
        :param p_ini: fraction of water hydrogen bonds
        :param n_kuhn: number of Kuhn lengths
        :param u_p: ratio of solvent to polymer volume fraction :math:`\\frac{\\upsilon_w}{\\upsilon_p}`
        :param u_a: ratio of salt to polymer volume fraction :math:`\\frac{\\upsilon_w}{\\upsilon_+}`
        :param u_b: ratio of salt to polymer volume fraction :math:`\\frac{\\upsilon_w}{\\upsilon_-}`
        :param f_a: math:`h_+ \\frac{\\upsilon_w}{\\upsilon_+} \\frac{\\phi_+}{\\(1 - self.phi_p - self.phi_+ - self.phi_-)}`
        :param f_b: math:`h_- \\frac{\\upsilon_w}{\\upsilon_-} \\frac{\\phi_-}{\\(1 - self.phi_p - self.phi_+ - self.phi_-)}`
        :param chi_p: a Flory Huggins parameter with the temperature dependence:math:`\\chi_p=A_p+\\frac{B_p}{T}`
        :param chi_e: a Flory Huggins parameter between polymer and electrolytes
        :param df_w: free energy change upon formation of hydrogen bond in water (in :math:`k_BT` units)
        :param df_p: free energy change upon formation of hydrogen bond in PEO (in :math:`k_BT` units)
        :param df_a: free energy change upon dissociated salt (in :math:`k_BT` units)
        :param df_b: free energy change upon dissociated salt (in :math:`k_BT` units)
        :param h_a: number of water molecules forming the hydration shell of the + ions
        :param h_b: number of water molecules forming the hydration shell of the - ions
        :param bjerrum_object: object of water class
        :param b_fac: B_factors for the different species in units of ion_size
        """

        self.phi_p = phi_p
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.n = n_kuhn
        self.u_p = u_p
        self.u_a = u_a
        self.u_b = u_b
        self.f_a = f_a
        self.f_b = f_b
        self.chi_p = chi_p
        self.chi_e = chi_e
        self.df_w = df_w
        self.df_p = df_p
        self.df_a = df_a
        self.df_b = df_b
        self.h_a = h_a
        self.h_b = h_b

        self.x = x
        self.p = p

        self.bjerrum_object = bjerrum_object
        self.b_fac = b_fac

        self.phi_w = 1 - self.phi_p - self.phi_a - self.phi_b
        self.phi_1 = self.phi_a + self.phi_b

        self.c = np.array([self.f_a / self.h_a * 1000 / 18, self.f_b / self.h_b * 1000 / 18])

    def dh_free(self, ion_size):
        """
        Debye Huckel Free energy according to :cite:`Levin1996`

        :param ion_size: Ion diameter (in Angstrom)
        :return: excess free energy per unit volume in units of :math:`\\frac{k_BT}{a^3}` (float)
        """

        dhfree = dh.DebyeHuckel(self.bjerrum_object).free_energy_db_excess(self.c, ion_size)
        return dhfree

    def hc_free(self, ion_size):
        """
        Excess free energy of the hard core

        :param c: concentration of the different species mols/litre (M)
        :param ion_size: ionic size (in Angstrom)
        :return: excess free energy (float)
        """

        hcfree = hc.HardCore(b_fac=self.b_fac).free_energy_hc_excess(self.c, ion_size)
        return hcfree

    def free(self, ion_size):
        """
        Free energy of a polymer in solution :math:`\\frac{F \\upsilon}{V k_B T}=\\frac{f}{k_B T}`
        (per unit volume in :math:`k_B T` units)

        :param ion_size: Ion diameter (in Angstrom)
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

        f_dh = self.dh_free(ion_size)
        f_hc = self.hc_free(ion_size)

        return f_ref_11 + f_ref_12 + f_ref_13 + f_ref_2 + f_int_1 + f_int_2 + f_as_1 + f_as_2 + f_as_3 + f_as_4 + f_as_5 + f_as_6 + f_as_7 + f_dh + f_hc


