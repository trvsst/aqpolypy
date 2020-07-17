"""
:module: PolymerSimplePEO
:platform: Unix, Windows, OS
:synopsis: Defines a simple Polyethylene Glycol class

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""
import numpy as np
import aqpolypy.polymer.PolymerPropertiesABC as pa


class PEOSimple(pa.PolymerProperties):

    """ Defines basic characteristic of a polymer"""

    def __init__(self, mw):
        """
        constructor

        :param mw: polymer molecular weight
        """

        m_unit = 44.052
        m_length = 3.64
        k_length = 7.24
        nu = 0.584

        super().__init__(mw, m_unit, m_length, k_length, nu)

        self.egy = self.hydrogen_bond_energy()
        self.enty = self.hydrogen_bond_entropy()
        self.ang = 0.0
        self.d_f = 0.0

        self.p_name = 'Polyethylene Oxide'

    def name(self):
        """
        polymer name

        :return: polymer name
        """

        return self.p_name

    def hydrogen_bond_energy(self, egy=1800):
        """
        Change in internal energy :math:`\\Delta E= E_{water} -  E_{peo}`

        :param egy: Internal energy (default is 1800 K)

        :return: Change in internal energy (in temperature units)
        """

        self.egy = egy

        return egy

    def hydrogen_bond_entropy(self, ang= np.pi/5.5):
        """
        Change in internal energy :math:`\\Delta S= S_{water} -  S_{peo}`

        :param ang: angle parameterizing the entropy (default is :math:`\\frac{5\\pi}{11)
        :return: Change in internal energy (in temperature units)
        """

        self.ang = ang

        self.enty = -np.log(0.5*(1-np.cos(ang)))

        return self.enty

    def hydrogen_bond_free_energy(self, temp):
        """

        Change in Free energy :math:`\\frac{\\Delta F}{k_BT}= \\frac{\\Delta E}{k_B T}-\\frac{\\Delta S}{k_B}`

        :param temp: Temperature in K
        :return: change in free energy
        """

        self.d_f = self.egy/temp - self.enty

        return self.df
