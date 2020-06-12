"""
:module: SaltPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added abstract methods: ionic_strength, molar_vol_infinite_dilution, density_sol,
..                                            molar_vol, osmotic_coeff, log_gamma.
"""

import numpy as np
from abc import ABC, abstractmethod


class SaltProperties(ABC):
    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature absolute
        :param pa: pressure in atm
        :instantiate: temperature, pressure, stoichiometry coefficients

        """
        # temperature and pressure
        self.tk = tk
        self.pa = pa

        # Calculations for stoichiometry coefficients
        self.mat_stoich = np.array([None])
        # nu_+ + nu_-
        self.nu = np.sum(self.mat_stoich[0])
        # nu_+ nu_-
        self.nu_prod = self.mat_stoich[0, 0] * self.mat_stoich[0, 1]
        # abs(z_+ z_-)
        self.z_prod = np.abs(self.mat_stoich[1, 0] * self.mat_stoich[1, 1])
        # nu_+, z_+
        self.nz_prod_plus = self.mat_stoich[0, 0] * self.mat_stoich[1, 0]

        self.mat = np.array([self.nu, self.nu_prod, self.z_prod, self.nz_prod_plus])

    @abstractmethod
    def ionic_strength(self, m):
        """
        Abstract method: calculates the ionic strength of electrolyte :math:
        """
        pass

    @abstractmethod
    def molar_vol_infinite_dilution(self):
        """
        Abstract method: calculates the ------- :math:
        """
        pass

    @abstractmethod
    def density_sol(self, m):
        """
        Abstract method: calculates the ------ :math:
        """
        pass

    @abstractmethod
    def molar_vol(self, m):
        """
        Abstract method: calculates the molar volume :math:
        """
        pass

    @abstractmethod
    def osmotic_coeff(self, m):
        """
        Abstract method: calculates the osmotic coefficient :math:
        """
        pass

    @abstractmethod
    def log_gamma(self, m):
        """
        Abstract method: calculates the ------ :math:
        """
        pass
