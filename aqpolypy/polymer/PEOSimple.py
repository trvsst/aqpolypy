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
        self.a_val = 0.0
        self.b_val = 0.0

        self.p_name = 'Polyethylene Oxide'

    def name(self):
        """
        polymer name

        :return: polymer name
        """

        return self.p_name

    def hydrogen_bond_energy(self, egy=2000):
        """
        Change in internal energy :math:`\\Delta E= E_{water} -  E_{peo}`

        :param egy: Internal energy (default is 2000 K)

        :return: Change in internal energy (in temperature units)
        """

        self.egy = egy

        return egy

    def hydrogen_bond_entropy(self, ang=np.pi/7):
        """
        Change in internal energy :math:`\\Delta S= S_{water} -  S_{peo}`

        parameterized by :math:`\\alpha`, :math:`\\Delta S = -\\log\\left(\\frac{1-\\cos(\\alpha)}{2}\\right)`

        :param ang: angle parameterizing the entropy (default is :math:`\\frac{\\pi}{7}`
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

        return self.d_f

    def chi(self, temp, a_val=-0.244, b_val=135):
        """
        Flory Huggins parameter :math:`\\chi(T)=A + \\frac{B}{T}`

        :param temp: temperature in K
        :param a_val: parameter A in Flory Huggins parameter
        :param b_val: parameter B in Flory Huggins parameter
        :return: Flory Huggins parameter for PEO water (float)
        """

        self.a_val = a_val
        self.b_val = b_val

        return self.a_val+self.b_val/temp
