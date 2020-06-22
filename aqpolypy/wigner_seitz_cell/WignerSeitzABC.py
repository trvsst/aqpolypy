"""
:module: WignerSeitzABC
:platform: Unix, Windows, OS
:synopsis: Defines an abstract unit class

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as integrate


class WiSe(ABC):

    """ Defines a general Wigner Seitz cell"""

    def __init__(self, dim, n_distinct_wedges):
        """
        constructor

        :param dim: plane is d=1, cylinder d=2, sphere d=3
        :param n_distinct_wedges: number of distinct wedges
        """

        self.dim = dim
        self.wedge = n_distinct_wedges*[]
        self.num_wedges = n_distinct_wedges*[]

    @abstractmethod
    def name(self):
        """
        name of the wigner seitz cell

        :return: name of the wigner-seitz cell
        """

    def s_angle(self):
        """
        Computes in d=2(3), the angle(solid angle) spanned ed by the wedge

        :return: angle, solid angle (float)
        """

        s_ang = 0.0
        if self.dim == 2:
            def fun(x):
                return 1
            for ind, n_w in enumerate(self.num_wedges):
                s_ang += n_w*integrate.quad(fun, self.wedge[ind][1], self.wedge[ind][0])
        elif self.dim == 3:
            def fun(x, y):
                return np.sin(y)
            for ind, n_w in enumerate(self.num_wedges):
                a, b = self.wedge[ind][0]
                f1, f2 = self.wedge[ind][1]
                s_ang += n_w*integrate.dblquad(fun, a, b, f1, f2)
        else:
            s_ang = 1

        return s_ang
