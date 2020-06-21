"""
:module: SaltPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added abstract methods: ionic_strength, molar_vol_infinite_dilution, density_sol,
..                                            molar_vol, osmotic_coeff, log_gamma.
..                  - Added stoichiometry coefficient to constructor.
"""

from abc import ABC, abstractmethod


class SaltProperties(ABC):
    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, stoichiometry coefficients

        """
        # temperature and pressure
        self.tk = tk
        self.pa = pa

        # Calculations for stoichiometry coefficients
        # ***** self.mat_stoich = np.array([None]) *****

    @abstractmethod
    def ionic_strength(self, m):
        """
        Abstract method: calculates the ionic strength of electrolyte :math:`I = \\frac{1}{2} \\sum_{i} m_i z_i^2`

        :param m: molality
        """
        pass

    @abstractmethod
    def molar_vol_infinite_dilution(self):
        """
        Abstract method: calculates the partial molal volume of solute at infinite dilution :math:`\\bar{\\upsilon}^{
        \\circ}_s`
        """
        pass

    @abstractmethod
    def density_sol(self, m):
        """
        Abstract method: calculates the density of electrolyte solution :math:`\\rho_{sol} = M_w
        \\frac{\\left(1 + \\frac{mM_2}{1000}\\right)}{\\upsilon_w + \\frac{mM_w\\upsilon_2}{1000}}`

        :param m: molality
        """
        pass

    @abstractmethod
    def molar_vol(self, m):
        """
        Abstract method: calculates the molar volume of electrolyte solution :math:`\\upsilon_s`

        :param m: molality
        """
        pass

    @abstractmethod
    def osmotic_coeff(self, m):
        """
        Abstract method: calculates the osmotic coefficient :math:`\\phi`

        :param m: molality
        """
        pass

    @abstractmethod
    def log_gamma(self, m):
        """
        Abstract method: calculates the activity coefficient :math:`\\ln\\gamma_{\\pm}`

        :param m: molality
        """
        pass
