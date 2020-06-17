"""
:module: PolymerPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving polymer classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

from abc import ABC, abstractmethod
import aqpolypy.units.units as un

class PolymerProperties(ABC):

    """ Defines basic characteristic of a polymer"""

    def __init__(self, mw, m_unit, m_length, k_length, nu):
        """
        constructor

        :param mw: polymer molecular weight
        :param m_unit: mass of a monomer unit
        :param m_length: length of a monomer unit
        :param k_length: Kuhn length
        :param nu: cross section linear factor
        """

        # molecular weight of the polymer
        self.mw = mw

        # molecular weight of one peo unit
        self.m_unit = m_unit

        # molecular length
        self.monomer_length = m_length

        # Kuhn length
        self.k_length = k_length

        # value of nu
        self.nu = nu

        # number of monomers
        self.n = mw / self.m_unit

        # ratio between real and kun
        self.ratio_real_kuhn = self.monomer_length/self.k_length

        # number of effective n
        self.n_p = self.n * self.ratio_real_kuhn

        # maximum length of the polymer
        self.max_length = self.k_length * self.n_p

        # maximum molecular area
        self.area = (self.nu * self.k_length) ** 2

        # volume of a kuhn monomer
        self.volume = self.area * self.k_length

        # number of renormalized monomers
        self.n_p = self.ratio_real_kuhn * self.n

        # density in kg/m^3
        num_dens = un.mol_angstrom_2_mol_mcube(1/self.volume)
        self.melt_density = 1e-3*self.m_unit*num_dens/self.ratio_real_kuhn

        # physical volume of a monomer
        self.vol = self.volume * self.ratio_real_kuhn
        # water molecular weight

    @abstractmethod
    def name(self):
        """
        Abstract method: polymer name
        """
        pass
