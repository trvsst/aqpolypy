"""
:module: BrushSolution
:platform: Unix, Windows, OS
:synopsis: Defines a Polymer Brush in Solvent

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

import numpy as np


class MakeBrushSolvent(object):

    """ Defines the free energy of a BrushSolvent System """

    def __init__(self, dim, chi, sigma, rad, pol, **kwargs):
        """
        constructor

        :param dim: plane is d=1, cylinder d=2, sphere d=3
        :param chi: Flory-Huggins Parameter :math:`\\chi`
        :param sigma: grafting density :math:`\\sigma`` in chains/nm :sup:`3`
        :param rad: nanocrystal radius in Angstrom
        :param pol: Polymer object
        :param kwargs: c_s and v_sol coefficients default to 2/3 and 29.91 (average water volume)
        :param c_s: coefficient of stretching energy
        :param v_sol: average volume of solvent molecule (default is water)
        """

        c_s = 2/3
        if 'c_s' in kwargs:
            c_s = kwargs['c_s']
        v_sol = 29.91
        if 'v_sol' in kwargs:
            v_sol = kwargs['v_sol']

        self.dim = dim
        self.chi = chi
        # convert to chains/A^2
        self.sigma = 1e-2*sigma
        self.rad = rad
        self.pol = pol
        self.c_s = c_s
        self.v_avg = v_sol

        self.solvent_good = True
        self.solvent_theta = False
        self.solvent_poor = False

        if self.chi == 0.5:
            self.solvent_theta = True
        if self.chi > 0.5:
            self.solvent_poor = True

        if self.solvent_theta:
            self.xi_t = np.inf
        else:
            self.xi_t = np.abs(1/(2*self.chi-1))

        # grafting density correlation length
        self.xi_s = 1/(self.pol.k_length*self.pol.nu*np.sqrt(self.sigma))

        # hat_r
        self.hat_r = self.rad / pol.k_length

        # ratio of monomer to solvent volumes
        self.r_vol = pol.volume / self.v_avg

        # free energy normalization
        self.f_norm = self.r_vol * self.xi_s ** 2

        # number of chains
        self.num_chains = self.sigma * np.pi * 8 * self.rad ** 3 / 6

        # n_p
        self.n_p = self.pol.n_p

        # define opm values
        self.param_lambda = self.n_p / self.hat_r
        self.tau = (1 + self.dim * self.param_lambda / self.xi_s ** 2) ** (1 / self.dim)
        self.opm_radius = self.hat_r*self.tau
        self.opm_radius_angstrom = self.opm_radius*pol.k_length
        self.opm_brush_radius = self.opm_radius - self.hat_r
        self.opm_brush_radius_angstrom = self.opm_brush_radius*pol.k_length

        # number of components
        self.num_of_components = None
