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

        super().__init__(dim, chi, sigma, rad, pol, **kwargs)

        # number of components is brush and solvent
        self.num_of_components = 2

        # lagrange multiplier enforcing the normalization condition
        self.lag = lag
        self.lag_ini = lag

    def f_ideal(self, phi):
        """
        Ideal free energy:

        .. math::
            :label: free_ideal

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
            :label: flory_huggins

            f_{fh}(\\phi)=\\frac{\\upsilon_p}{\\upsilon}\\hat{\\xi}_S^2\\left[\\chi \\phi(u)(1-\\phi(u))\\right]

        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the free energy (float)
        """

        return self.f_norm * self.chi * phi * (1 - phi)

    def f_stretch(self, phi):
        """
        Stretching free energy

        .. math::
            :label: stretch

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
            :label: geom

            Q(u)=\\left(1+\\frac{u}{\\hat{R}}\\right)^{d-1}

        :param u: variable :math:`u\\equiv \\frac{z}{b}`, where z is the perpendicular direction
        :return: value of the factor (float)
        """

        return (1 + u / self.hat_r) ** (self.dim - 1)

    def f_dens(self, u, phi):
        """
        free energy density, is given from Eq. :eq:`free_ideal`, :eq:`flory_huggins`, :eq:`stretch` and :eq:`geom`

        .. math::
            :label: free_density

            f(u,\\phi) = Q(u)\\left(f_{id}(\\phi)+f_{fg}(\\phi)\\right)+\\frac{f_s(\\phi)}{Q(u)}

        :param u: variable :math:`u\\equiv\\frac{z}{b}`, where z is the perpendicular direction
        :param phi: polymer volume fraction  :math:`\\phi(u)`

        :return: value of free energy (float)
        """

        val0 = self.intg(u)

        val1 = self.f_ideal(phi) + self.f_flory_huggins(phi)
        val2 = self.f_stretch(phi)

        return val0 * val1 + val2 / val0

    def free_energy(self, size_b=None, phi_b=None):
        """
        free energy of the brush: integration of Eq. :eq:`free_density` over the brush size Eq. :eq:`brush_size`

        .. math::
            :label: free

            f(H)\\equiv\\frac{F}{{\\cal N} k_B T}=\\int_0^{H} du f(u,\\phi(u))

        :param size_b: brush size
        :param phi_b: function phi
        :return: value of the free energy (float)
        """

        if size_b is None:
            h_m = self.determine_h()
        else:
            h_m = size_b

        if phi_b is None:
            phi_a = self.phi
        else:
            phi_a = phi_b

        def fun(u):
            return self.f_dens(u, phi_a(u))

        return integrate.quad(fun, 0.0, h_m)

    def lhs_eqn_phi(self, phi):
        """
        Equation of the minimization for the ideal and Flory-Huggins

        .. math::
            :label: E_def

            E_1(\\phi(u)) \\equiv \\frac{\\upsilon}{\\upsilon_p}\\frac{1}{N}\\log\\left(\\frac{\\phi(u)}{N}\\right)
            -\\log(1-\\phi(u))+\\chi(1-2\\phi(u))+\\Lambda

        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the quantity (float)
        """

        ct2_1 = np.log(phi / self.n_p) / (self.n_p * self.r_vol)
        ct2_2 = -np.log(1 - phi)
        ct2_3 = -2*self.chi*phi

        return ct2_1 + ct2_2 + ct2_3 + self.lag + self.chi

    def eqn_min_phi(self, u, phi):
        """
        Equation satisfied by :math:`\\phi(u)`. It is

        .. math::
            :label: min_equation

            E_1(\\phi(u))-\\frac{1}{\\hat{\\xi}^4_S}\\frac{1}{c_s}\\frac{\\upsilon}{\\upsilon_p}
            \\frac{1}{\\phi^2(u)(1+\\frac{u}{\\hat{R}})^{2d-2}}=0

        :math:`E_1` is defined by Eq. :eq:`E_def`.

        :param u: variable :math:`u\\equiv\\frac{z}{b}`, where z is the perpendicular direction
        :param phi: polymer volume fraction :math:`\\phi(u)`
        :return: value of the equation
        """

        val1 = self.lhs_eqn_phi(phi)
        val2 = self.f_stretch(phi) / (self.f_norm * phi * (self.intg(u)) ** 2)

        return val1 - val2

    def determine_lagrange(self, h, lag=None):
        """
        Determine the value of the Lagrange parameter :math:`\\Lambda` that gives a brush of size :math:`h`

        :param h: value of brush size (in Angstrom)
        :param lag: initial value of Lagrange multiplier
        """

        # brush size in dimensionless units
        h_var = h/self.pol.k_length

        if lag is None:
            lag = 1e-3 - self.chi

        def intgr(u):
            return self.intg(u) * self.phi(u)

        def der_intgr(u):
            return self.intg(u) * self.der_phi_lag(u)

        def fun(x):
            self.lag = x
            return integrate.quad(intgr, 0, h_var)[0] - self.n_p / self.xi_s ** 2

        def der_fun(x):
            self.lag = x
            return integrate.quad(der_intgr, 0, h_var)[0]

        return optimize.newton(fun, lag, der_fun)

    def determine_h(self):
        """
        Compute the size of the brush for a given Lagrange parameter :math:`\\Lambda`

        .. math::
            :label: brush_size

            H(\\Lambda)

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
        function

        .. math:: \\phi(u)
            :label: phi

        obtained by solving Eq. :eq:`min_equation`. Note that it verifies the normalization Eq. :eq:`norm`

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

    def der_phi_lag(self, u):
        """
        computes the function

        .. math::
            :label: phi_der

            \\frac{\\partial \\phi(u)}{\\partial \\Lambda}=-\\frac{\\phi(u)(1-\\phi(u))}{\\frac{\\upsilon}{\\upsilon_p}
            \\frac{1}{N}(1-\\phi(u))\\left(1+2\\log(\\frac{\\phi(u)}{N}\\right)+\\phi(u)-6\\chi\\phi(u)(1-\\phi(u))
            +2(1-\\phi(u))\\left(\\Lambda+\\chi-\\log(1-\\phi(u))\\right)}

        :param u: variable :math:`u\\equiv\\frac{z}{b}`, where z is the perpendicular direction
        :return: value of :math:`\\frac{\\partial \\phi(u)}{\\partial \\Lambda}` (float)
        """

        z = self.phi(u)

        c1 = (1-z)*(1+2*np.log(z/self.n_p))/(self.r_vol*self.n_p)
        c2 = z- 6*self.chi*z*(1-z)+2*(1-z)*(self.lag+self.chi-np.log(1-z))

        return -z*(1-z)/(c1+c2)

    def optimal_lambda(self):
        """
        Computes the optimal value of :math:`\\Lambda`, defined as the value that
        makes the free energy a minimum with respect the brush length :math:`H`, that is:

        .. math::
            :label: f_der_min

            \\frac{\\partial}{\\partial H} f(H)=0

        see Eq. :eq:`free`

        :return: optimal :math:`\\Lambda``  ( OptimizeResultsObject_ )

        .. _OptimizeResultsObject: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """

        def fopt(lam):
            self.lag = lam
            return self.free_energy()[0]

        return optimize.minimize_scalar(fopt)

    def der_free_energy(self):
        """
        Computes the derivative of the free energy:

        .. math::
            :label: f_der

            \\frac{\\partial f}{\\partial H}

        see Eq. :eq:`free`

        :return: derivative of the free energy (float)
        """

        h_t = self.determine_h()
        phi_h = self.phi(h_t)

        return self.f_dens(h_t, phi_h) + self.f_norm * self.intg(h_t) * self.lag * phi_h

    def inv_phi(self, phi):
        """
        Computes the inverse function :math:`\\phi` Eq. :eq:`phi`

        .. math::
            :label: u_phi

            u(\\phi) \\rightarrow  u(\\phi(u))=u

        :param phi: polymer volume fraction  :math:`\\phi(u)`
        :return: value of :math:`u(\\phi)` (float)
        """

        val = self.f_norm * self.lhs_eqn_phi(phi) * phi / (self.f_stretch(phi))

        return self.hat_r * (val ** (-0.5 / (self.dim - 1)) - 1)

    def phi_normalization(self):
        """
        verifies that the :math:`\\phi(u)``  is properly normalized, namely

        .. math::
            :label: norm

            \\frac{\\xi_S^2}{N} \\int_0^{\\hat{H}}du\\phi(u) = 1

        :return:  (normalization, error) as a tuple
        """

        def intgr(u):
            return self.intg(u) * self.phi(u)

        fac = self.xi_s ** 2 / self.n_p

        val = integrate.quad(intgr, 0.0, self.determine_h())

        return [fac * val[0], fac * val[1]]
