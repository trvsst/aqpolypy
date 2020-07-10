"""
:module: salt_nacl_rp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for SaltNaClRP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added tests for all methods in SaltPropertiesPitzer
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.salt.SaltNaClRP as nacl


class TestSaltNaClRP(unittest.TestCase):

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
        salt_nacl = nacl.NaClPropertiesRogersPitzer(300)
        test_vals = np.allclose(salt_nacl.h_fun(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_nacl = nacl.NaClPropertiesRogersPitzer(300)
        test_vals = np.allclose(salt_nacl.h_fun_gamma(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_nacl = nacl.NaClPropertiesRogersPitzer(300)
        test_vals = np.allclose(salt_nacl.p_fun_gamma(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_params_der_p(self):
        # pressure derivative parameters in [temperature (C), pressure (bar), [bp, 2cp]]
        param = np.array([[10, 1, 1.956e-5, -2.25e-6],
                          [30, 1, 1.069e-5, -1.07e-6],
                          [90, 1, 2.577e-6, -6.02e-8],
                          [40, 400, 6.779e-6, -7.19e-7],
                          [50, 800, 4.436e-6, -4.71e-7]])
        # converting to [temperature (K), pressure (atm), [bp, cp]]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 3] = param[:, 3] / 2
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals1 = np.allclose(salt_nacl.params_der_p[1], param[:, 2], 0, 1e-2)
        test_vals2 = np.allclose(salt_nacl.params_der_p[2], param[:, 3], 0, 1e-2)
        self.assertTrue(test_vals1)
        self.assertTrue(test_vals2)

    def test_stoichiometry_coeffs(self):
        # parameters in stoichiometry coefficient[nu = 2, nu_prod = 1, z_prod = 1, nz_prod_plus = 1]
        param = np.array([2, 1, 1, 1])
        salt_nacl = nacl.NaClPropertiesRogersPitzer(300)
        test_vals = np.allclose(salt_nacl.mat, param, 0, 1e-6)
        self.assertTrue(test_vals)

    def test_ionic_strength(self):
        # parameters in [molality mol/kg, ionic strength]
        param = np.array([[0.1, 0.1],
                          [0.25, 0.25],
                          [0.50, 0.50],
                          [0.75, 0.75],
                          [1, 1],
                          [2, 2],
                          [3, 3]])
        # testing params up to a precision of 10^-6
        salt_nacl = nacl.NaClPropertiesRogersPitzer(300)
        test_vals = np.allclose(salt_nacl.ionic_strength(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_molar_vol_infinite_dilution(self):
        # parameters in [temperature (C), pressure (bar), partial molal volume of solute cm^3/mol]
        param = np.array([[10, 1, 1.506e1],
                          [50, 1, 1.774e1],
                          [90, 1, 1.710e1],
                          [20, 400, 1.789e1],
                          [50, 600, 1.967e1]])
        # converting to [temperature (K), pressure (atm), partial molal volume of solute m^3/mol]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 2] = param[:, 2] / 1e6
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_nacl.molar_vol_infinite_dilution(), param[:, 2], 0, 1e-2)
        self.assertTrue(test_vals)

    def test_density_sol(self):
        # parameters in [temperature (C), pressure (bar), molality, specific volume of NaCl(aq) cm^3/g]
        param = np.array([[10, 1, 1, 0.961101],
                          [10, 1, 0.1, 0.995998],
                          [10, 1, 0.5, 0.979804],
                          [40, 1, 2, 0.938287],
                          [40, 1, 3, 0.910145],
                          [40, 1, 0.75, 0.979243],
                          [30, 200, 1, 0.959139],
                          [60, 200, 3, 0.9129],
                          [40, 400, 0.1, 0.987267],
                          [70, 600, 0.25, 0.9883]])
        # converting to [temperature (K), pressure (atm), molality, specific volume of NaCl(aq) kg/m^3]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 3] = (1 / param[:, 3]) * 1e3
        # testing params up to a precision of 10^-1
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_nacl.density_sol(param[:, 2]), param[:, 3], 0, 1e-1)
        self.assertTrue(test_vals)

    def test_molar_vol(self):
        # parameters in [temperature (C), pressure (bar), molality, molar volume cm^3/mol]
        param = np.array([[10, 1, 1, 17.00347],
                          [10, 1, 0.1, 15.59673],
                          [10, 1, 0.5, 16.34880],
                          [40, 1, 2, 20.05013],
                          [40, 1, 3, 20.61311],
                          [40, 1, 0.75, 19.10164],
                          [30, 200, 1, 19.49070],
                          [60, 200, 3, 21.52742],
                          [40, 400, 1, 20.52720],
                          [70, 800, 2, 22.34101]])
        # converting to [temperature (K), pressure (atm), molality, molar volume m^3/mol]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 3] = param[:, 3] / 1e6
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_nacl.molar_vol(param[:, 2]), param[:, 3], 0, 1e-2)
        self.assertTrue(test_vals)

    def test_osmotic_coeff(self):
        # parameters in [temperature (C), molality, osmotic coefficient]
        param = np.array([[5, 1, 0.92436938],
                          [5, 0.1, 0.93242304],
                          [5, 0.5, 0.9160134],
                          [25, 1, 0.93588145],
                          [25, 2, 0.98430138],
                          [45, 3, 1.05708439],
                          [45, 0.7, 0.92870989],
                          [65, 3, 1.05824243],
                          [85, 0.1, 0.92489648],
                          [95, 0.2, 0.9140571]])

        # converting to [temperature (K), molality, osmotic coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0])
        test_vals = np.allclose(salt_nacl.osmotic_coeff(param[:, 1]), param[:, 2], 0, 1e-2)
        self.assertTrue(test_vals)

    def test_log_gamma(self):
        # parameters in [temperature (C), molality, activity coefficient]
        param = np.array([[5, 1, -0.43818367],
                          [5, 0.1, -0.24821997],
                          [5, 0.5, -0.39019683],
                          [25, 1, -0.42229531],
                          [25, 2, -0.40443826],
                          [45, 3, -0.32152333],
                          [45, 0.7, -0.41035063],
                          [65, 3, -0.33051837],
                          [85, 0.1, -0.28213002],
                          [95, 0.2, -0.35491577]])
        # converting to [temperature (K), molality, activity coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0])
        test_vals = np.allclose(salt_nacl.log_gamma(param[:, 1]), param[:, 2], 0, 1e-2)
        self.assertTrue(test_vals)

    def test_apparent_molal_enthalpy(self):
        # parameters in [temperature (C), molality, apparent relative molal enthalpy (cal/mol)]
        param = np.array([[0, 1.0, -188.11417381],
                          [0, 0.1, 32.99636565],
                          [0, 0.5, -55.55049346],
                          [25, 1.0, 4.22338911],
                          [25, 2.0, -134.37285678],
                          [25, 3.0, -252.47092572],
                          [25, 1.2, -24.15561616],
                          [50, 3.5, 135.16767688],
                          [50, 0.1, 148.41561978],
                          [75, 0.8, 411.78485441]])
        # converting to [temperature (K), molality, apparent relative molal enthalpy (cal/mol)]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0])
        test_vals = np.allclose(un.joule_2_cal(salt_nacl.apparent_molal_enthalpy(param[:, 1])), param[:, 2], 0, 1e-2)
        self.assertTrue(test_vals, "TEST VALUES DO NOT COME FROM LITERATURE")


if __name__ == '__main__':
    unittest.main()
