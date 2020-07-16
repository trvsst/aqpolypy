"""
:module: PolymerSolution
:platform: Unix, Windows, OS
:synopsis: Defines a Polymer in Solvent, including hydrogen bonds

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, July2020
.. history:
..
"""
import numpy as np
from scipy.special import xlogy as lg
from scipy.optimize import fsolve


class PolymerSolution(object):
    """
    Defines a solution with polymers with hydrogen bonds following the model :cite:`Dormidontova2002`
    """

    def __init__(self, phi, n_kuhn, f_v, chi, df_w, df_p):
        """
        The constructor

        :param phi: polymer fraction :math:`\\phi`
        :param n_kuhn: number of Kuhn lengths
        :param x: fraction of polymer hydrogen bonds
        :param p: fraction of water hydrogen bonds
        :param f_v: ratio of volume fractions :math:`\\frac{\\upsilon}{\\upsilon_p}`
        :param chi: Flory-Huggins parameter :math:`\\chi`
        :param df_w: free energy change upon formation of hydrogen bond in water
        :param df_p: free energy change upon formation of hydrogen bond in PEO
        """

        self.phi = phi
        self.n = n_kuhn
        self.f_v = f_v
        self.chi = chi
        self.df_w = df_w
        self.df_p = df_p

    def free(self, x, p):
        """
        Free energy of a polymer in solution

        :param x: fraction of polymer hydrogen bonds
        :param p: fraction of water hydrogen bonds

        :return: value of free energy (float)
        """

        f_ref_1 = lg(self.f_v * self.phi / self.n, self.phi / (self.n * np.exp(1)))
        f_ref_2 = lg((1 - self.phi), (1 - self.phi) / np.exp(1))

        f_int = self.chi * self.phi * (1 - self.phi)

        f_as_1 = 2 * self.phi * self.f_v * (lg(x, x) + lg(1 - x, 1 - x) - x * self.df_p)
        f_as_2 = 2 * (1 - self.phi) * (lg(p, p) + lg(1 - p, 1 - p) - p * self.df_w)

        z_val = 1 - p - x * self.phi * self.f_v / (1 - self.phi)

        f_as_3 = 2 * (1 - self.phi) * lg(z_val, z_val)
        f_as_4 = -2 * (1 - self.phi) * lg(1 - z_val, (2 * (1 - self.phi) / np.exp(1)))

        return f_ref_1 + f_ref_2 + f_int + f_as_1 + f_as_2 + f_as_3 + f_as_4

    def eqns(self, val):
        """
        Equations determining the fraction of hydrogen bonds

        :param val: ndarray containing x,p
        :return: equations (ndarray)
        """

        x, p = val

        fac = 2 * (1 - self.phi) * (1 - p - x * self.phi / (self.f_v * (1 - self.phi)))

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
