"""
:module: water_wagner_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for WaterWagner

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Created test for ideal gas part and residual part
..                  - Created test for residual part delta derivative
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.water.WaterWagner as ww


class TestWaterWagner(unittest.TestCase, ww.WaterWagner):

    # Testing ideal gas part
    def test_phi_naught(self):
        # [density, temperature(K), phi_naught] Wagner TABLE 6.6
        param = np.array([[838.025, 500, 2.04797734],
                          [358, 647, -1.56319605]])

        # testing phi_naught up to a precision of 10^-8
        water = ww.WaterWagner(param[:, 0], param[:, 1])
        test_val = np.allclose(water.phi_naught(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_val)

    # Testing residual part
    def test_phi_r(self):
        # [density, temperature(K), phi_r] Wagner TABLE 6.6
        param = np.array([[838.025, 500, -3.42693206],
                          [358, 647, -1.21202657]])

        # testing phi_r up to a precision of 10^-8
        water = ww.WaterWagner(param[:, 0], param[:, 1])
        test_val = np.allclose(water.phi_r(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_val)

    # Testing residual part delta derivative
    def test_phi_r_der_del(self):
        # [density, temperature(K), phi_r_der_del] Wagner TABLE 6.6
        param = np.array([[838.025, 500, -0.364366650],
                          [358, 647, -0.714012024]])

        # testing phi_r_der_del up to a precision of 10^-8
        water = ww.WaterWagner(param[:, 0], param[:, 1])
        test_val = np.allclose(water.phi_r_der_del(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_val)


if __name__ == '__main__':
    unittest.main()
