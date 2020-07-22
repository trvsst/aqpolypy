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
        test_vals = np.allclose(salt_nacl.p_fun_gamma(2, param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_params_der_p(self):
        # pressure derivative parameters in [temperature (C), pressure (bar), [BV, 2CV]]
        param = np.array([[70, 1, 3.513e-6, -1.22e-7],
                          [60, 1, 4.415e-6, -2.00e-7],
                          [90, 1, 2.577e-6, -6.02e-8],
                          [80, 400, 2.552e-6, -7.97e-8],
                          [60, 800, 3.062e-6, -2.00e-7]])
        # converting to [temperature (K), pressure (atm), [bp, cp]]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 3] = param[:, 3] / 2
        # testing params up to a precision of 10^-9
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals1 = np.allclose(salt_nacl.params_der_p[1], param[:, 2], 0, 1e-9)
        test_vals2 = np.allclose(salt_nacl.params_der_p[4], param[:, 3], 0, 1e-9)
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
        param = np.array([[70, 1, 1.781e1],
                          [60, 1, 1.791e1],
                          [90, 1, 1.710e1],
                          [80, 400, 1.884e1],
                          [60, 800, 2.024e1]])
        # converting to [temperature (K), pressure (atm), partial molal volume of solute m^3/mol]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 2] = param[:, 2] / 1e6
        # testing params up to a precision of 10^-8
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_nacl.molar_vol_infinite_dilution(), param[:, 2], 0, 1e-8)
        self.assertTrue(test_vals)

    def test_density_sol(self):
        # parameters in [temperature (C), pressure (bar), molality, specific volume of NaCl(aq) cm^3/g]
        param = np.array([[60, 1, 1, 0.9797],
                          [60, 1, 0.1, 1.0130],
                          [60, 1, 0.5, 0.9976],
                          [80, 1, 2, 0.9581],
                          [80, 1, 3, 0.9293],
                          [80, 1, 0.75, 0.9999],
                          [70, 200, 1, 0.9772],
                          [90, 200, 3, 0.9280],
                          [80, 400, 0.1, 1.0074],
                          [90, 600, 0.25, 0.9998]])
        # converting to [temperature (K), pressure (atm), molality, specific volume of NaCl(aq) cm^3/g]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        # testing params up to a precision of 10^-4
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose((1 / salt_nacl.density_sol(param[:, 2])) * 1e3, param[:, 3], 0, 1e-4)
        self.assertTrue(test_vals)

    def test_osmotic_coeff(self):
        # parameters in [temperature (C), molality, osmotic coefficient]
        param = np.array([[0, 1, 0.920],
                          [0, 0.1, 0.932],
                          [0, 0.5, 0.914],
                          [25, 1, 0.936],
                          [25, 2, 0.985],
                          [50, 3, 1.059],
                          [50, 0.7, 0.929],
                          [75, 3, 1.056],
                          [75, 0.1, 0.927],
                          [100, 0.2, 0.913]])
        # converting to [temperature (K), molality, osmotic coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-3
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0])
        test_vals = np.allclose(salt_nacl.osmotic_coeff(param[:, 1]), param[:, 2], 0, 1e-3)
        self.assertTrue(test_vals)

    def test_log_gamma(self):
        # parameters in [temperature (C), molality, activity coefficient]
        param = np.array([[0, 1, 0.640],
                          [0, 0.1, 0.780],
                          [0, 0.5, 0.675],
                          [25, 1, 0.656],
                          [25, 2, 0.668],
                          [50, 3, 0.726],
                          [50, 0.7, 0.662],
                          [75, 3, 0.710],
                          [75, 0.1, 0.759],
                          [100, 0.2, 0.698]])
        # converting to [temperature (K), molality, activity coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0])
        test_vals = np.allclose(np.exp(salt_nacl.log_gamma(param[:, 1])), param[:, 2], 0, 1e-2)
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
