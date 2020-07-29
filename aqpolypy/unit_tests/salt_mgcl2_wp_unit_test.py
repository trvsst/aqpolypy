"""
:module: salt_mgcl2_wp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for SaltMgCl2WP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added tests for all methods in SaltModelPitzer
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.salt.SaltMgCl2WP as mgcl2


class TestSaltMgCl2WP(unittest.TestCase):

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
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(300)
        test_vals = np.allclose(salt_mgcl2.h_fun(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(300)
        test_vals = np.allclose(salt_mgcl2.h_fun_gamma(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(300)
        test_vals = np.allclose(salt_mgcl2.p_fun_gamma(2, param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_stoichiometry_coeffs(self):
        # parameters in stoichiometry coefficient[nu = 3, nu_prod = 2, z_prod = 2, nz_prod_plus = 2]
        param = np.array([3, 2, 2, 2])
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(300)
        test_vals = np.allclose(salt_mgcl2.mat, param, 0, 1e-6)
        self.assertTrue(test_vals, salt_mgcl2.mat)

    def test_ionic_strength(self):
        # parameters in [molality mol/kg, ionic strength]
        param = np.array([[0.1, 0.3],
                          [0.25, 0.75],
                          [0.50, 1.50],
                          [0.75, 2.25],
                          [1, 3],
                          [2, 6],
                          [3, 9]])
        # testing params up to a precision of 10^-6
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(300)
        test_vals = np.allclose(salt_mgcl2.ionic_strength(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_molar_vol(self):
        # parameters in [pressure (Pa), molality, molar volume cm^3/mol]
        param = np.array([[2e6, 0.01, 10.09],
                          [2e6, 0.10, 11.58],
                          [2e6, 0.80, 14.60],
                          [2e6, 1.40, 17.41],
                          [10e6, 0.10, 11.61],
                          [10e6, 0.80, 14.51],
                          [10e6, 1.40, 17.43],
                          [20e6, 0.10, 12.07],
                          [20e6, 0.80, 14.83],
                          [20e6, 1.40, 17.89]])
        # converting to [pressure (atm), molality, molar volume m^3/mol]
        param[:, 0] = param[:, 0] / un.atm_2_pascal(1)
        param[:, 2] = param[:, 2] / 1e6
        # testing params up to a precision of 10^-8
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(373.15, param[:, 0])
        test_vals = np.allclose(salt_mgcl2.molar_vol(param[:, 1]), param[:, 2], 0, 1e-8)
        self.assertTrue(test_vals, str(salt_mgcl2.molar_vol(param[:, 1])) + " & " + str(param[:, 2]))

    def test_osmotic_coeff(self):
        # parameters in [molality, osmotic coefficient]
        param = np.array([[0.001, 0.9560],
                          [0.005, 0.9161],
                          [0.01, 0.8942],
                          [0.02, 0.8717],
                          [0.05, 0.8460],
                          [0.10, 0.8332],
                          [0.20, 0.8289],
                          [0.30, 0.8344],
                          [0.40, 0.8461],
                          [0.50, 0.8628]])
        # testing params up to a precision of 10^-4
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(373.15)
        test_vals = np.allclose(salt_mgcl2.osmotic_coeff(param[:, 0]), param[:, 1], 0, 1e-4)
        self.assertTrue(test_vals, str(salt_mgcl2.osmotic_coeff(param[:, 0])) + " & " + str(param[:, 1]))

    def test_log_gamma(self):
        # parameters in [molality, activity coefficient]
        param = np.array([[0.001, -0.1382],
                          [0.005, -0.2786],
                          [0.01, -0.3661],
                          [0.02, -0.4697],
                          [0.05, -0.6253],
                          [0.10, -0.7498],
                          [0.20, -0.8718],
                          [0.30, -0.9349],
                          [0.40, -0.9694],
                          [0.50, -0.9853]])
        # testing params up to a precision of 10^-4
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(373.15)
        test_vals = np.allclose(salt_mgcl2.log_gamma(param[:, 0]), param[:, 1], 0, 1e-4)
        self.assertTrue(test_vals, str(salt_mgcl2.log_gamma(param[:, 0])) + " & " + str(param[:, 1]))

    def test_apparent_molal_enthalpy(self):
        # parameters in [molality, apparent relative molal enthalpy (kJ/mol)]
        param = np.array([[0.001, 0.81],
                          [0.005, 1.65],
                          [0.01, 2.20],
                          [0.02, 2.87],
                          [0.05, 3.96],
                          [0.10, 4.97],
                          [0.20, 6.23],
                          [0.30, 7.18],
                          [0.40, 7.99],
                          [0.50, 8.72]])
        # testing params up to a precision of 10^-2
        salt_mgcl2 = mgcl2.MgCl2PropertiesWangPitzer(373.15, 5e6 / un.atm_2_pascal(1))
        test_vals = np.allclose(salt_mgcl2.apparent_molal_enthalpy(param[:, 0]) / 1e3, param[:, 1], 0, 1e-2)
        self.assertTrue(test_vals, str(salt_mgcl2.apparent_molal_enthalpy(param[:, 0]) / 1e3) + " & " + str(param[:, 1]))


if __name__ == '__main__':
    unittest.main()
