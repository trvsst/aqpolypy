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
        Abstract method: calculates the partial molal volume of solute at infinite dilution :math:
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
        Abstract method: calculates the molar volume of electrolyte solution :math:

        :param m: molality
        """
        pass

    @abstractmethod
    def osmotic_coeff(self, m):
        """
        Abstract method: calculates the osmotic coefficient :math:`\\phi = 1 - |z_Mz_X|A_\\phi\\frac{I^{\\frac{1}{2}}}
        {1+bI^{\\frac{1}{2}}}+m\\frac{2\\nu_M\\nu_X}{\\nu}\\left(\\beta^{(0)}_{MX}\\beta^{(1)}_{MX}e^{-\\alpha
        I^{\\frac{1}{2}}}\\right)+m^{2}\\frac{2\\left(\\nu_M\\nu_X\\right)^{\\frac{3}{2}}}{\\nu}C^{\\phi}_{MX}`

        :param m: molality
        """
        pass

    @abstractmethod
    def log_gamma(self, m):
        """
        Abstract method: calculates the activity coefficient :math:`\\ln\\gamma_{\\pm} = -|z_Mz_X|A_{\\phi}
        \\left(\\frac{I^{\\frac{1}{2}}}{1+ bI^{\\frac{1}{2}}}+\\frac{2}{b}\\ln\\left(1+bI^{\\frac{1}{2}}\\right)\\right)
        +m\\frac{2\\nu_M\\nu_X}{\\nu}\\left(2\\beta^{(0)}_{MX}+\\frac{2\\beta^{(1)}_{MX}}{\\alpha^2I}
        \\left[1-\\left(1+\\alpha I^{\\frac{1}{2}}-\\frac{\\alpha^2I}{2}\\right)e^{-\\alpha I^{\\frac{1}{2}}}
        \\right]\\right)+\\frac{3m^2}{2}\\left(\\frac{2\\left(\\nu_M\\nu_X\\right)}{\\nu}C^{\\phi}_{MX}\\right)`

        :param m: molality
        """
        pass
