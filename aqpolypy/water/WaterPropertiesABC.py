"""
:module: WaterPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added molecular weight of water to constructor.
..                  - Made abstract methods: density, molar_volume, dielectric_constant, compressibility.
..                  - Added temperature and pressure parameters to constructor.
..                  - Added water polarizability and dipole moment to constructor.
..                  - Made abstract methods: a_phi and a_v
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
        Abstract method: calculates the density of water
        """
        pass

    @abstractmethod
    def molar_volume(self):
        """
        Abstract method: calculates the molar volume
        """
        pass

    @abstractmethod
    def dielectric_constant(self):
        """
        Abstract method: calculates the dielectric constant
        """
        pass

    @abstractmethod
    def compressibility(self):
        """
        Abstract method: calculates the compressibility of water
        """
        pass

    @abstractmethod
    def a_phi(self):
        """
        Abstract method: calculates the osmotic coefficient of water
        """
        pass

    @abstractmethod
    def a_v(self):
        """
        Abstract method: calculates the apparent molal volume of water
        """
        pass
