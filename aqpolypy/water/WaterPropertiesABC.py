"""
:module: WaterPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added molecular weight of water to constructor.
..                  - Made abstract methods: density, molar_volume, dielectric_constant, compressibility.
..                  - Added temperature and pressure parameters to constructor.
..                  - Added water polarizability and dipole moment to constructor.
..                  - Made abstract methods: a_phi and a_v.
..                  - Added enthalpy_coefficient abstract method.
"""


from abc import ABC, abstractmethod


class WaterProperties(ABC):
    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: temperature, pressure, molecular weight, polarizability, dipole moment

        """

        # water molecular weight
        self.MolecularWeight = 18.01534
        # water polarizability
        self.alpha = 18.1458392e-30
        # water dipole moment
        self.mu = 6.1375776e-30
        # temperature and pressure
        self.tk = tk
        self.pa = pa

    @abstractmethod
    def density(self):
        """
        Abstract method: calculates the density of water :math:`\\rho_w`
        """
        pass

    @abstractmethod
    def molar_volume(self):
        """
        Abstract method: calculates the molar volume :math:`\\upsilon_w`
        """
        pass

    @abstractmethod
    def dielectric_constant(self):
        """
        Abstract method: calculates the relative dielectric constant :math:`\\varepsilon_{r}`
        """
        pass

    @abstractmethod
    def compressibility(self):
        """
        Abstract method: calculates the isothermal compressibility of water

        :math:`\\beta_T = -\\frac{1}{V}\\left(\\frac{\\partial V}{\\partial P} \\right)_T`
        """
        pass

    @abstractmethod
    def a_phi(self):
        """
        Abstract method: calculates the osmotic coefficient of water defined by

        :math:`A_{\\phi}=\\frac{1}{3}\\left(\\frac{2\\pi N_A\\rho_w l_B}{10^3}\\right)^{1/2}
        \\left(\\frac{e^2}{4\\pi \\varepsilon_0 \\varepsilon_r}\\right)^{3/2}`

        dimensionless because it is normalized to reference molality of :math:`\\sqrt{m^{\\ominus}}`
        with :math:`m^{\\ominus}=\\mbox{kg mol}^{-1}`
        """
        pass

    @abstractmethod
    def a_v(self):
        """
        Abstract method: calculates the apparent molal volume of water defined by

        :math:`A_{v}=-2A_{\\phi}RT[3\\left(\\frac{\\partial\\varepsilon}{\\partial P}+\\beta_w\\right)`
        """
        pass

    @abstractmethod
    def enthalpy_coefficient(self):
        """
        Abstract method: calculates the enthalpy coefficient of water defined by

        :math:`A_{H}=-9A_{\\phi}\\left[1+T\\left(\\frac{\\partial \\ln D}{\\partial T}\\right)_{P}+\\frac{T\\alpha_{w}}{3}\\right]`
        """