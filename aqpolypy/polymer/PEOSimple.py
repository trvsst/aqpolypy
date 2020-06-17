"""
:module: PolymerSimplePEO
:platform: Unix, Windows, OS
:synopsis: Defines a simple Polyethylene Glycol class

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

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

        self.p_name = 'Polyethylene Oxide'

    def name(self):
        """
        polymer name

        :return: polymer name
        """

        return self.p_name
