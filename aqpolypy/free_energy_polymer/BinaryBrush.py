"""
:module: SphericalBrush
:platform: Unix, Windows, OS
:synopsis: Defines a Spherical Brush in Solvent

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

import numpy as np
from scipy.special import xlogy as lg
import scipy.integrate as integrate
from scipy import optimize
from aqpolypy.free_energy_polymer.BrushSolution import MakeBrushSolvent


class BinaryBrush(MakeBrushSolvent):

    """ Defines the free energy of a BrushSolvent System """

    def __init__(self, dim, chi, sigma, rad, pol, lag, **kwargs):
        """
        constructor

        :param dim: plane is d=1, cylinder d=2, sphere d=3
        :param chi: Flory-Huggins Parameter :math:`\\chi`
        :param sigma: grafting density :math:`\\sigma`` in chains/nm :sup:`3`
        :param rad: nanocrystal radius in Angstrom
        :param pol: Polymer object
        :param lag: Initial Lagrange Multiplier :math:`\\Lambda` (necessary to enforce normalization of :math:`\\phi`)
        :param kwargs: optional arguments c_s, v_solvent
        """

        super().__init__(dim, chi, sigma, rad, pol, kwargs)

        # number of components is brush and solvent
        self.num_of_components = 2

        # free energy normalization
        self.f_norm = self.r_vol * self.xi_s ** 2

        # lagrange multiplier enforcing the normalization condition
        self.lag = lag
        self.lag_ini = lag

    def f_ideal(self, phi):
        """
        Ideal free energy:

        .. math::

            f_{id}(\\phi)=\\frac{\\upsilon_p}{\\upsilon}\\hat{\\xi}_S^2\\left[\\frac{\\upsilon}{\\upsilon_p}
            \\frac{\\phi(u)}{N}\\log\\left(\\frac{\\phi(u)}{Ne}\\right)+(1-\\phi(u))\\log\\left(\\frac{1-\\phi(u)}{e}
            \\right)+1\\right]

        :param phi: polymer volume fraction :math:`\\phi(u)`

        :return: value of the free energy (float)
        """

        id_1 = lg(phi / (self.r_vol * self.n_p), phi / (self.n_p * np.exp(1)))
        id_2 = lg(1 - phi, (1 - phi) / np.exp(1)) + 1

        return self.f_norm * (id_1 + id_2)

    def f_flory_huggins(self, phi):
        """
        Flory Huggins free energy:

        .. math::

            f_{fh}(\\phi)=\\frac{\\upsilon_p}{\\upsilon}\\hat{\\xi}_S^2\\left[\\chi \\phi(u)(1-\\phi(u))\\right]

        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the free energy (float)
        """

        return self.f_norm * self.chi * phi * (1 - phi)

    def f_stretch(self, phi):
        """
        Stretching free energy

        .. math::

            f_{s}(\\phi)=\\frac{\\upsilon_p}{\\upsilon}\\hat{\\xi}_S^2\\left[\\frac{1}{\\hat{\\xi}^4_S}
            \\frac{1}{c_s}\\frac{\\upsilon}{\\upsilon_p}\\frac{1}{\\phi(u)}\\right]

        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the free energy (float)
        """

        return self.f_norm / ((self.xi_s ** 4) * self.c_s * self.r_vol * phi)

    def intg(self, u):
        """
        Geometric factor

        .. math::

            Q(u)=\\left(1+\\frac{u}{\\hat{R}}\\right)^{d-1}

        :param u: variable :math:`u\\equiv \\frac{z}{b}`, where z is the perpendicular direction
        :return: value of the factor (float)
        """

        return (1 + u / self.hat_r) ** (self.dim - 1)

    def f(self, u, phi):
        """
        free energy density, given as

        .. math::

            f(u,\\phi) = Q(u)\\left(f_{id}(\\phi)+f_{fg}(\\phi)\\right)+\\frac{f_s(\\phi)}{Q(u)}

        :param u: variable :math:`u\\equiv\\frac{z}{b}`, where z is the perpendicular direction
        :param phi: polymer volume fraction  :math:`\\phi(u)`

        :return: value of free energy (float)
        """

        val0 = self.intg(u)

        val1 = self.f_ideal(phi) + self.f_flory_huggins(phi)
        val2 = self.f_stretch(phi)

        return val0 * val1 + val2 / val0

    def free_energy(self):
        """
        free energy of the Brush

        .. math::
            f\\equiv\\frac{F}{{\\cal N} k_B T}=\\int_0^{H} du f(u,\\phi(u))

        :return: value of the free energy (float)
        """

        h_m = self.determine_h()

        def fun(u):
            return self.f(u, self.phi(u))

        return integrate.quad(fun, 0.0, h_m)

    def lhs_eqn_phi(self, phi):
        """
        Equation of the minimization for the ideal and Flory-Huggins

        .. math::
            E_1(\\phi) \\equiv \\frac{\\upsilon}{\\upsilon_p}\\frac{1}{N}\\log\\left(\\frac{\\phi(u)}{N}\\right)
            -\\log(1-\\phi(u))+\\chi(1-2\\phi(u))+\\Lambda

        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the quantity (float)
        """

        ct2_1 = np.log(phi / self.n_p) / (self.n_p * self.r_vol)
        ct2_2 = -(phi + np.log(1 - phi))
        ct2_3 = -phi / self.xi_t

        return ct2_1 + ct2_2 + ct2_3 + self.lag

    def eqn_min_phi(self, u, phi):
        """
        Equation satisfied by :math:`\\phi(u)`. It is

        .. math::
            E_1(\\phi)-\\frac{1}{\\hat{\\xi}^4_S}\\frac{1}{c_s}\\frac{\\upsilon}{\\upsilon_p}
            \\frac{1}{\\phi^2(u)(1+\\frac{u}{\\hat{R}})^{2d-2}}=0

        :param u: variable :math:`u\\equiv\\frac{z}{b}`, where z is the perpendicular direction
        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the equation
        """

        val1 = self.lhs_eqn_phi(phi)
        val2 = self.f_stretch(phi) / (self.f_norm * phi * (self.intg(u)) ** 2)

        return val1 - val2

    def determine_h(self):
        """
        Compute the size of the brush :math:`H`, (for fixed :math:`\\Lambda`)

        :return: optimal the value of :math:`H` (float)
        """

        def intgr(u):
            return self.intg(u) * self.phi(u)

        def fun(h):
            return integrate.quad(intgr, 0, h)[0] - self.n_p / self.xi_s ** 2

        x0 = 0.0

        return optimize.newton(fun, x0, intgr)

    def phi(self, u, tol=1e-7):
        """
        function :math:`\\phi(u)`

        :param u: variable :math:`u\\equiv\\frac{z}{b}`, where z is the perpendicular direction
        :param tol: tolerance
        :return: value of :math:`\\phi(u)` (float)
        """

        def fp_phi(x):
            """
            equation to solve

            :return : value of function (float)
            """

            return self.eqn_min_phi(u, x)

        ic = [tol, 1 - tol]

        return optimize.brentq(fp_phi, ic[0], ic[1])

    def optimal_lambda(self):
        """

        Computes the optimal value of :math:`\\Lambda`, defined as the value that
        makes the free energy a minimum with respect the brush length :math:`H`, that is:

        .. math::
            \\frac{\\partial}{\partial H} f=0

        :return: optimal :math:`\\Lambda``
        """

        def fopt(lam):
            self.lam = lam
            return self.free_energy()[0]

        return optimize.minimize_scalar(fopt)
