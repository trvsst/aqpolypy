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
    def __init__(self, p, t):
        """
        constructor

        :param p: pressure in atmospheres
        :param t: temperature in celsius
        :instantiate: molecular weight, pressure, and temperature
        :itype : float
        """
        self.MolecularWeight = 18.01534
        self.pressure = p
        self.temperature = t


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
