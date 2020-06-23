"""
:module: RhombicDodecahedron
:platform: Unix, Windows, OS
:synopsis: Defines the fcc Wigner-Seitz Cell

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

import aqpolypy.wigner_seitz_cell.WignerSeitzABC as Ws
import numpy as np


class RhombicDodecahedron(Ws.WiSe):

    """ Defines a general Wigner Seitz cell"""

    def __init__(self):
        """
        constructor

        """

        dimension = 3
        num_distinct_wedges = 1

        super().__init__(dimension, num_distinct_wedges)

        psi = np.pi/6
        delta = np.arctan(1/np.sqrt(2))

        def f1(x):
            return 0

        def f2(x):
            return np.arctan(np.tan(psi)/np.cos(x-delta))

        self.wedges[0] = ((0, np.pi/2), (f1, f2))
        self.num_wedges[0] = 48

    def max_theta(self):
        """
        maximum value of the angle :math:`\\theta` for the corresponding wedge

        :return: maximum value of the angle :math:`\\theta` (ndarray)
        """
        return np.array([np.pi/4])

    def __str__(self):
        """
        String representation

        :return: name of the class
        """

        return 'fcc Wigner-Seitz cell (Rhombic Dodecahedron)'
