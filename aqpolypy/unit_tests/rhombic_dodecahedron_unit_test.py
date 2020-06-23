"""
:module: rhombic_dodecahedron_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for rhombic dodecahedron class

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""
import unittest
import numpy as np
import aqpolypy.wigner_seitz_cell.RhombicDodecahedron as RoD


class TestRhombicDodecahedron(unittest.TestCase):
    """
    Test the different functions
    """

    def test_area(self):
        # tests the area
        rho = RoD.RhombicDodecahedron()

        # check solid angle
        ang = rho.s_angle()
        ang_total = rho.s_angle_tot()
        # result is 4pi for the total angle, 4pi/48 num_wedges = 120
        res = 4*np.pi
        self.assertLess(np.abs(ang_total-res), 1e-9)
        self.assertLess(np.abs(ang[0] - res/rho.num_wedges[0]), 1e-9)

        # maximum value of theta
        ang = np.pi/4
        self.assertLess(np.abs(rho.max_theta()[0]-ang), 1e-9)

        # Check the volume
        dn = 1
        vl = rho.vol(dn)
        vl_tot = rho.vol_tot(dn)
        # edge size
        a_val = dn*3/np.sqrt(6)
        # icosahedron volume
        vl_exact = 16*np.sqrt(3)*a_val**3/9
        vl_wedge = vl_exact/48
        self.assertLess(np.abs(vl[0]-vl_wedge), 1e-9)
        self.assertLess(np.abs(vl_tot-vl_exact), 1e-9)


if __name__ == '__main__':
    unittest.main()
