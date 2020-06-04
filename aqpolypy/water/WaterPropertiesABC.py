"""
:module: WaterPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - describe changes
"""


from abc import ABC, abstractmethod


class WaterProperties(ABC):
    def __init__(self, tk, pa=1):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :instantiate: molecular weight, temperature, and pressure
        :itype : float
        """
        self.MolecularWeight = 18.01534
        self.alpha = 18.1458392e-30
        self.mu = 6.1375776e-30
        self.tk = tk
        self.pa = pa

    @abstractmethod
    def density(self):
        """
        Abstract method: calculates the density of water
        """
        pass

    def molar_volume(self):
        """
        Abstract method: calculates the molar volume
        """
        pass

    def dielectric_constant(self):
        """
        Abstract method: calculates the dielectric constant
        """
        pass

    def compressibility(self):
        """
        Abstract method: calculates the compressibility of water
        """
        pass
