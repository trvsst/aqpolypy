"""
:module: salt_mgso4_pp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for SaltMgSO4PP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added tests for all methods in SaltModelPitzer
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.salt.SaltMgSO4PP as mgso4


class TestSaltMgSO4PP(unittest.TestCase):

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
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_mgso4.h_fun(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_mgso4.h_fun_gamma(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_mgso4.p_fun_gamma(2, param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_stoichiometry_coeffs(self):
        # parameters in stoichiometry coefficient[nu = 2, nu_prod = 1, z_prod = 4, nz_prod_plus = 2]
        param = np.array([2, 1, 4, 2])
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_mgso4.mat, param, 0, 1e-6)
        self.assertTrue(test_vals)

    def test_ionic_strength(self):
        # parameters in [molality mol/kg, ionic strength]
        param = np.array([[0.1, 0.4],
                          [0.25, 1.0],
                          [0.50, 2.0],
                          [0.75, 3.0],
                          [1, 4],
                          [2, 8],
                          [3, 12]])
        # testing params up to a precision of 10^-6
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(300)
        test_vals = np.allclose(salt_mgso4.ionic_strength(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_density_sol(self):
        # parameters in [temperature (K), pressure (bar), molality, density of solution g/cm^3]
        param = np.array([[323.90, 22.3, 0.0723, 0.99720],
                          [324.48, 22.3, 0.0723, 0.99699],
                          [324.24, 49.1, 0.0723, 0.99819],
                          [324.26, 98.3, 0.0723, 1.00020],
                          [294.21, 21.9, 0.1693, 1.01927],
                          [294.23, 23.1, 0.1693, 1.01924],
                          [293.84, 47.6, 0.1693, 1.02043],
                          [324.76, 47.5, 0.1693, 1.00917],
                          [325.27, 95.6, 0.1693, 1.01083],
                          [294.90, 95.2, 0.2503, 1.03156]])
        # converting to [temperature (K), pressure (atm), molality, density of solution g/cm^3]
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        # testing params up to a precision of 10^-5
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose((salt_mgso4.density_sol(param[:, 2])) / 1e3, param[:, 3], 0, 1e-5)
        self.assertTrue(test_vals, str((salt_mgso4.density_sol(param[:, 2])) * 1e3) + " & " + str(param[:, 3]))

    def test_molar_vol(self):
        # parameters in [temperature (C), pressure (bar), molality, molar volume cm^3/mol]
        param = np.array([[0.0, 1.0, 0.05, -7.39],
                          [0.0, 1.0, 0.20, -3.72],
                          [0.0, 100, 0.25, -1.14],
                          [25.0, 1.0, 0.05, -2.16],
                          [25.0, 1.0, 0.20, 0.76],
                          [25.0, 100, 0.25, 3.27],
                          [60.0, 1.0, 0.05, -1.23],
                          [60.0, 1.0, 0.20, 1.17],
                          [60.0, 100, 0.25, 3.68]])
        # converting to [temperature (C), pressure (atm), molality, molar volume m^3/mol]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_pascal(1)
        param[:, 3] = param[:, 3] / 1e6
        # testing params up to a precision of 10^-8
        salt_mgso4 = mgso4.MgSO4PropertiesPhutelaPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_mgso4.molar_vol(param[:, 2]), param[:, 3], 0, 1e-8)
        self.assertTrue(test_vals, str(salt_mgso4.molar_vol(param[:, 2])) + " & " + str(param[:, 3]))


if __name__ == '__main__':
    unittest.main()
