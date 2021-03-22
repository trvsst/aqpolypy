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
        # [temperature(K), density, phi_naught] Wagner TABLE 6.6
        param = np.array([[500, 838.025, 2.04797734],
                          [647, 358, -1.56319605]])

        # testing phi_naught up to a precision of 10^-8
        water = ww.WaterWagner(param[:, 0], d=param[:, 1])
        test_val = np.allclose(water.phi_naught(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_val)

    # Testing residual part
    def test_phi_r(self):
        # [temperature(K), density, phi_r] Wagner TABLE 6.6
        param = np.array([[500, 838.025, -3.42693206],
                          [647, 358, -1.21202657]])

        # testing phi_r up to a precision of 10^-8
        water = ww.WaterWagner(param[:, 0], d=param[:, 1])
        test_val = np.allclose(water.phi_r(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_val)

    # Testing density
    def test_density(self):
        # [temperature (K), pressure (MPa), density (SI)] Wagner TABLE 13.2
        param = np.array([[280, 0.05, 999.886],
                          [310, 0.05, 993.361],
                          [340, 0.05, 979.513],
                          [290, 0.1, 998.803],
                          [315, 0.1, 991.496],
                          [350, 0.1, 973.728],
                          [275, 0.101325, 999.938],
                          [320, 0.101325, 989.427],
                          [360, 0.101325, 967.404],
                          [295, 0.25, 997.875],
                          [360, 0.25, 967.471],
                          [390, 0.25, 945.658],
                          [300, 0.50, 996.736],
                          [380, 0.50, 953.505],
                          [410, 0.50, 929.012],
                          [285, 0.75, 999.824],
                          [330, 0.75, 985.07],
                          [370, 0.75, 960.894],
                          [275, 1, 1000.39],
                          [300, 1, 996.96],
                          [390, 1, 946.029],
                          [285, 2, 1000.41],
                          [340, 2, 980.37],
                          [380, 2, 954.222],
                          [295, 3, 999.123],
                          [335, 3, 983.499],
                          [360, 3, 968.717],
                          [280, 4, 1001.8],
                          [310, 4, 995.094],
                          [370, 4, 962.398],
                          [290, 5, 1001.06],
                          [370, 5, 962.858],
                          [390, 5, 947.993],
                          [280, 6, 1002.76],
                          [320, 6, 991.987],
                          [370, 6, 963.317],
                          [280, 7, 1003.24],
                          [335, 7, 985.232],
                          [400, 7, 940.919],
                          [300, 8, 1000.07],
                          [380, 8, 957.056],
                          [420, 8, 924.104],
                          [275, 10, 1004.85],
                          [340, 10, 983.84],
                          [370, 10, 965.139],
                          [370, 50, 982.351],
                          [270, 400, 1140.15],
                          [1273, 1000, 809.28]])

        # converting parameters to [temperature (K), pressure (atm), density (SI)]
        param[:, 1] = (param[:, 1] * 1000000) / un.atm_2_pascal(1)
        """
        for i in param:
            # testing density up to a precision of 10^-2
            water = ww.WaterWagner(i[0], i[1])
            test_val = np.allclose(water.density_brentq(), i[2], 0, 1e-2)
            self.assertTrue(test_val, msg=water.density_brentq())
        """
        for i in param:
            # testing density up to a precision of 10^-2
            water = ww.WaterWagner(i[0], i[1])
            test_val = np.allclose(water.density_fsolve(), i[2], 0, 1e-2)
            self.assertTrue(test_val)

    # Testing residual part delta derivative
    def test_phi_r_der_del(self):
        # [temperature(K), density, phi_r_der_del] Wagner TABLE 6.6
        param = np.array([[500, 838.025, -0.364366650],
                          [647, 358, -0.714012024]])

        # testing phi_r_der_del up to a precision of 10^-8
        water = ww.WaterWagner(param[:, 0], d=param[:, 1])
        test_val = np.allclose(water.phi_r_der_del(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_val)


if __name__ == '__main__':
    unittest.main()
