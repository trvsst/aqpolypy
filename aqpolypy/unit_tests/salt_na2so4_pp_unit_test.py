"""
:module: salt_na2so4_pp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for SaltNa2SO4PP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added tests for all methods in SaltModelPitzer
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.salt.SaltNa2SO4PP as na2so4


class TestSaltNa2SO4PP(unittest.TestCase):

    def test_h_fun(self):
        # parameters in [ionic strength, h_fun]
        param = np.array([[0.1, 0.1340424057],
                          [0.25, 0.1958348455],
                          [0.50, 0.2559957171],
                          [0.75, 0.296905218],
                          [1, 0.3285239002],
                          [2, 0.413400379],
                          [3, 0.4685124111]])
        # testing params up to a precision of 10^-6
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_na2so4.h_fun(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_h_fun_gamma(self):
        # parameters in [ionic strength, h_fun_gamma]
        param = np.array([[0.1, 0.765407667],
                          [0.25, 1.095839382],
                          [0.50, 1.406507087],
                          [0.75, 1.612303325],
                          [1, 1.768641055],
                          [2, 2.177956004],
                          [3, 2.436684943]])
        # testing params up to a precision of 10^-6
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_na2so4.h_fun_gamma(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_p_fun_gamma(self):
        # parameters in [ionic strength, p_fun_gamma]
        param = np.array([[0.1, 0.5973924753],
                          [0.25, 0.4481808382],
                          [0.50, 0.3280905085],
                          [0.75, 0.260674695],
                          [1, 0.2161661792],
                          [2, 0.1262676179],
                          [3, 0.08733961077]])
        # testing params up to a precision of 10^-6
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_na2so4.p_fun_gamma(2, param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_stoichiometry_coeffs(self):
        # parameters in stoichiometry coefficient[nu = 3, nu_prod = 2, z_prod = 2, nz_prod_plus = 2]
        param = np.array([3, 2, 2, 2])
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_na2so4.mat, param, 0, 1e-6)
        self.assertTrue(test_vals)

    def test_ionic_strength(self):
        # parameters in [molality mol/kg, ionic strength]
        param = np.array([[0.1, 0.3],
                          [0.25, 0.75],
                          [0.50, 1.5],
                          [0.75, 2.25],
                          [1, 3],
                          [2, 6],
                          [3, 9]])
        # testing params up to a precision of 10^-6
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_na2so4.ionic_strength(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_density_sol(self):
        # parameters in [temperature (K), pressure (bar), molality, density of solution g/cm^3]
        param = np.array([[294.87, 20.2, 0.0578, 1.00608],
                          [325.16, 20.4, 0.0578, 0.99503],
                          [325.09, 20.5, 0.0578, 0.99507],
                          [295.18, 20.3, 0.0578, 1.00604],
                          [295.67, 48.8, 0.0578, 1.00717],
                          [325.01, 49.1, 0.0578, 0.99634],
                          [295.47, 95.3, 0.0578, 1.00928],
                          [325.08, 96.6, 0.0578, 0.99833],
                          [296.35, 20.0, 0.1085, 1.01207],
                          [294.85, 47.6, 0.1085, 1.01366]])
        # converting to [temperature (K), pressure (atm), molality, density of solution g/cm^3]
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        # testing params up to a precision of 10^-5
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose((salt_na2so4.density_sol(param[:, 2])) * 1e3, param[:, 3], 0, 1e-5)
        self.assertTrue(test_vals)

    def test_molar_vol(self):
        # parameters in [temperature (C), pressure (bar), molality, molar volume cm^3/mol]
        param = np.array([[0.0, 1.0, 0.05, 4.94],
                          [0.0, 1.0, 0.20, 8.61],
                          [0.0, 100, 0.30, 12.06],
                          [25.0, 1.0, 0.05, 13.89],
                          [25.0, 1.0, 0.20, 16.81],
                          [25.0, 100, 0.30, 19.78],
                          [60.0, 1.0, 0.05, 17.35],
                          [60.0, 1.0, 0.20, 19.90],
                          [60.0, 100, 0.30, 22.62]])
        # converting to [temperature (C), pressure (atm), molality, molar volume m^3/mol]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_pascal(1)
        param[:, 3] = param[:, 3] / 1e6
        # testing params up to a precision of 10^-8
        salt_na2so4 = na2so4.Na2SO4PropertiesPhutelaPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_na2so4.molar_vol(param[:, 2]), param[:, 3], 0, 1e-8)
        self.assertTrue(test_vals, str(salt_na2so4.molar_vol(param[:, 2])) + " & " + str(param[:, 3]))


if __name__ == '__main__':
    unittest.main()
