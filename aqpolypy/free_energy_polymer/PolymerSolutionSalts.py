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
from scipy.optimize import fsolve

import aqpolypy.salts_theory.DebyeHuckel as dh
import aqpolypy.salts_theory.HardCore as hc

class PolymerSolutionSalts(object):
    """
    Defines a solution with polymers with hydrogen bonds following the model :cite:`Dormidontova2002`
    """

    def __init__(self, phi_p, phi_a, phi_b, x_ini, p_ini, n_kuhn, u_p, u_a, u_b, f_a, f_b, chi_p, chi_e, df_w, df_p, df_a, df_b, h_a, h_b):
        """
        The constructor

        :param phi_p: polymer fraction :math:`\\phi_p`
        :param phi_a: salt fraction :math:`\\phi_a`
        :param phi_b: salt fraction :math:`\\phi_b`
        :param x_ini: initial fraction of polymer hydrogen bonds
        :param p_ini: initial fraction of water hydrogen bonds
        :param n_kuhn: number of Kuhn lengths
        :param u_p: ratio of solvent to polymer volume fraction :math:`\\frac{\\upsilon_w}{\\upsilon_p}`
        :param u_a: ratio of salt to polymer volume fraction :math:`\\frac{\\upsilon_w}{\\upsilon_+}`
        :param u_b: ratio of salt to polymer volume fraction :math:`\\frac{\\upsilon_w}{\\upsilon_-}`
        :param f_a: math:`h_a \\frac{\\upsilon_w}{\\upsilon_+} \\frac{\\phi_+}{\\( 1 - self.phi_p - self.phi_a - self.phi_b )}`
        :param f_b: math:`h_b \\frac{\\upsilon_w}{\\upsilon_-} \\frac{\\phi_-}{\\( 1 - self.phi_p - self.phi_a - self.phi_b )}`
        :param chi_p: Flory-Huggins parameter with the temperature dependence:math:`\\chi_p=A_p+\\frac{B_p}{T}` 
        :param chi_e: a Flory Huggins parameter between polymer and electrolytes 
        :param df_w: free energy change upon formation of hydrogen bond in water (in :math:`k_BT` units)
        :param df_p: free energy change upon formation of hydrogen bond in PEO (in :math:`k_BT` units)
        :param df_a: free energy change upon dissociated salt (in :math:`k_BT` units)
        :param df_b: free energy change upon dissociated salt (in :math:`k_BT` units)
        :param h_a: number of water molecules forming the hydration shell of the + ions 
        :param h_b: number of water molecules forming the hydration shell of the - ions         
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
        

        
        self.x, self.y = self.solv_eqns(x_ini, p_ini)
        
       

    def free(self, c, ion_size, phi_w, phi_1):
        """
        Free energy of a polymer in solution :math:`\\frac{F \\upsilon}{V k_B T}=\\frac{f}{k_B T}`
        (per unit volume in :math:`k_B T` units)

	:param c: concentration of ions (both + and -) mols/litre (M)
        :param ion_size: Ion diameter (in Angstrom)
        :param phi_w: math:`1 - \\phi_p - \\phi_+ - \\phi_-`
        :param phi_1: math:`\\phi_+ + \\phi_-`
        :return: value of free energy (float)
        """

        x = self.x
        p = self.y
        
        self.c = c
        self.ion_size = ion_size
        
        self.phi_w = 1 - self.phi_p - self.phi_a - self.phi_b
        self.phi_1 = self.phi_a + self.phi_b
        
        f_ref_11 = lg(self.u_p * self.phi_p / self.n, self.phi_p / (self.n * np.exp(1)) )
        f_ref_12 = lg(self.u_a * self.phi_a, self.phi_a /  np.exp(1))
        f_ref_13 = lg(self.u_b * self.phi_b, self.phi_b / np.exp(1))
        
        f_ref_2 = lg(self.phi_w, self.phi_w / np.exp(1))

        f_int_1 = self.chi_p * self.phi_p * self.phi_w
        f_int_2 = self.chi_e * self.phi_p * self.phi_1
                

        f_as_1 = 2 * self.u_p * self.phi_p * (lg(x, x) + lg(1 - x, 1 - x) - x * self.df_p)
        f_as_2 = 2 * self.phi_w * (lg(p, p) + lg(1 - self.f_a - p, 1 - self.f_a - p) - p * self.df_w)

        z_val = (1 - self.f_b - p - x * self.u_p * self.phi_p) / self.phi_w
        
        f_as_3 = 2 * self.phi_w * lg(z_val, z_val)
        f_as_4 = -2 * self.phi_w * lg(x * self.u_p * self.phi_p / self.phi_w + p, 2 * self.phi_w / np.exp(1))
        
        f_as_5 = self.phi_w * ( (np.log(np.math.factorial(h_a)) - df_a) * f_a + (np.log(np.math.factorial(h_b)) - df_b) * f_b ) 
        f_as_6 = self.phi_w * ( lg(1 - f_a -f_b, 1 - f_a -f_b) + lg(1 - f_a, 1 - f_a) + lg(1 - f_b, 1 - f_b) )
        f_as_7 = self.phi_w * (f_a +f_b) * np.log(self.phi_w / np.exp(1))

        return f_ref_11 + f_ref_12 + f_ref_13 + f_ref_2 + f_int1 + f_int2 + f_as_1 + f_as_2 + f_as_3 + f_as_4 + f_as_5 + f_as_6 + f_as_7 + dh.DebyeHuckel(self.bjerrum_object).free_energy_db_excess(self.c, self.ion_size) + hc.HardCore(self.b_fac).free_energy_db_excess(self.c, self.ion_size)



    def eqns(self, val):
        """
        Equations determining the fraction of hydrogen bonds

        :param val: ndarray containing x,p
        :return: equations (ndarray)
        """

        x, p = val
        

        fac = 2 * ( 1 - self.phi_p - self.phi_a - self.phi_b ) * (1 - p - x * self.u_p * (self.phi_p + ( self.phi_a + self.phi_b )) / ( 1 - self.phi_p - self.phi_a - self.phi_b ) )

        eqn1 = np.exp(self.df_p) * (1 - x) * fac
        eqn2 = np.exp(self.df_w) * (1 - p) * fac

        return np.array([x - eqn1, p - eqn2])

    def solv_eqns(self, x, p):
        """
        Solution to the equations defining the fraction of hydrogen bonds

        :param x: initial value for fraction of polymer hydrogen bonds
        :param p: initial value for fraction of water hydrogen bonds
        :return: number of hydrogen bonds ( OptimizeResultsObject_ )

        .. _OptimizeResultsObject: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """

        sol = fsolve(self.eqns, np.array([x, p]))

        return sol
