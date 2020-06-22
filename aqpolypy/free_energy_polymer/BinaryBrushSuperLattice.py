"""
:module: BinaryBrushSuperLattice
:platform: Unix, Windows, OS
:synopsis: Defines a Spherical Brush in Solvent

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

import numpy as np
import scipy.integrate as integrate
from scipy import optimize
from aqpolypy.free_energy_polymer.BrushSolution import MakeBrushSolvent


class BinaryBrushSuperLattice(MakeBrushSolvent):

    """ Defines the free energy of a Brush in a superlattice"""

    def __init__(self, dim, chi, sigma, rad, pol, d_n, **kwargs):
        """
        constructor

        :param dim: plane is d=1, cylinder d=2, sphere d=3
        :param chi: Flory-Huggins Parameter :math:`\\chi`
        :param sigma: grafting density :math:`\\sigma`` in chains/nm :sup:`3`
        :param rad: nanocrystal radius in Angstrom
        :param pol: Polymer object
        :param d_n: nearest neighbor distance for nanocrystals within the lattice in Angstrom
        :param kwargs: optional arguments c_s, v_solvent
        """

        super().__init__(dim, chi, sigma, rad, pol, **kwargs)

        # number of components is brush and solvent
        self.num_of_components = 2

        # compute the maximum and minimum values of the brush size
        self.d_n = d_n
        self.h_min = 0.5*self.d_n/pol.k_length - self.hat_r
        self.h_max = None
