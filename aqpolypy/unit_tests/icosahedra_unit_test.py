"""
:module: icosahedra_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for icosahedra class

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..
"""
import unittest
import numpy as np
import aqpolypy.wigner_seitz_cell.Icosahedra as Ic


class TestIcosahedra(unittest.TestCase):
    """
    Test the different functions
    """

    def test_area(self):
        # tests the area
        ico = Ic.Icosahedra()

        # let us compute solid angle
        ang = ico.s_angle()
        ang_total = ico.s_angle_tot()
        # result is 4pi for the total angle, 4pi/120 num_wedges = 120
        res = 4*np.pi
        self.assertLess(np.abs(ang_total-res), 1e-9)
        self.assertLess(np.abs(ang[0] - res/ico.num_wedges[0]), 1e-9)

        # Computing the volume
        dn = 1
        vl = ico.vol(dn)
        vl_tot = ico.vol_tot(dn)
        # edge size
        a_val = dn*12/(np.sqrt(3)*(3+np.sqrt(5)))
        # icosahedron volume
        vl_exact = 5*(3+np.sqrt(5))*a_val**3/12
        # volume of the wedge computed with Mathematica
        vl_wedge = 1/np.sqrt(6*(47+21*np.sqrt(5)))
        self.assertLess(np.abs(vl[0]-vl_wedge), 1e-9)
        self.assertLess(np.abs(vl_tot-vl_exact), 1e-9)


if __name__ == '__main__':
    unittest.main()
