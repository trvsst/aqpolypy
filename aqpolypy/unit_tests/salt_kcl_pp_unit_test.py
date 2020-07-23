"""
:module: salt_kcl_pp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for SaltNaClRP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added tests for all methods in SaltModelPitzer
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.salt.SaltKClPP as kcl


class TestSaltKClPP(unittest.TestCase):

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
        salt_kcl = kcl.KClPropertiesPabalanPitzer(300)
        test_vals = np.allclose(salt_kcl.h_fun(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_kcl = kcl.KClPropertiesPabalanPitzer(300)
        test_vals = np.allclose(salt_kcl.h_fun_gamma(param[:, 0]), param[:, 1], 0, 1e-6)
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
        salt_kcl = kcl.KClPropertiesPabalanPitzer(300)
        test_vals = np.allclose(salt_kcl.p_fun_gamma(2, param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_stoichiometry_coeffs(self):
        # parameters in stoichiometry coefficient[nu = 2, nu_prod = 1, z_prod = 1, nz_prod_plus = 1]
        param = np.array([2, 1, 1, 1])
        salt_kcl = kcl.KClPropertiesPabalanPitzer(300)
        test_vals = np.allclose(salt_kcl.mat, param, 0, 1e-6)
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
        salt_kcl = kcl.KClPropertiesPabalanPitzer(300)
        test_vals = np.allclose(salt_kcl.ionic_strength(param[:, 0]), param[:, 1], 0, 1e-6)
        self.assertTrue(test_vals)

    def test_molar_vol(self):
        # parameters in [temperature (C), pressure (bar), molality, molar volume cm^3/mol]
        param = np.array([[0, 196.62, 0.3331, 26.280],
                          [0, 196.62, 0.5009, 26.573],
                          [0, 196.62, 0.6693, 26.856],
                          [0, 196.62, 0.8360, 26.976],
                          [25, 196.62, 0.3331, 28.873],
                          [25, 196.62, 0.5009, 29.104],
                          [25, 196.62, 0.6693, 29.258],
                          [50, 196.62, 0.3331, 29.757],
                          [50, 196.62, 0.5009, 29.968],
                          [50, 196.62, 0.6693, 30.125]])
        # converting to [temperature (K), pressure (atm), molality, molar volume m^3/mol]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 3] = param[:, 3] / 1e6
        # testing params up to a precision of 10^-9
        salt_kcl = kcl.KClPropertiesPabalanPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_kcl.molar_vol(param[:, 2]), param[:, 3], 0, 1e-9)
        self.assertTrue(test_vals, str(salt_kcl.molar_vol(param[:, 2])) + " & " + str(param[:, 3]))

    def test_osmotic_coeff(self):
        # parameters in [temperature (C), molality, osmotic coefficient]
        param = np.array([[25, 1, 0.899],
                          [25, 0.1, 0.927],
                          [25, 0.5, 0.901],
                          [25, 0.3, 0.907],
                          [25, 2, 0.914],
                          [50, 3, 0.954],
                          [50, 0.5, 0.903]])

        # converting to [temperature (K), molality, osmotic coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-3
        salt_kcl = kcl.KClPropertiesPabalanPitzer(param[:, 0], 1 / un.atm_2_bar(1))
        test_vals = np.allclose(salt_kcl.osmotic_coeff(param[:, 1]), param[:, 2], 0, 1e-3)
        self.assertTrue(test_vals, str(salt_kcl.osmotic_coeff(param[:, 1])) + " & " + str(param[:, 2]))

    def test_log_gamma(self):
        # parameters in [temperature (C), molality, activity coefficient]
        param = np.array([[25, 1, 0.605],
                          [25, 0.1, 0.768],
                          [25, 0.5, 0.650],
                          [25, 0.3, 0.687],
                          [25, 2.0, 0.574],
                          [50, 3.0, 0.584],
                          [50, 0.5, 0.646]])
        # converting to [temperature (K), molality, activity coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-3
        salt_kcl = kcl.KClPropertiesPabalanPitzer(param[:, 0], 1 / un.atm_2_bar(1))
        test_vals = np.allclose(np.exp(salt_kcl.log_gamma(param[:, 1])), param[:, 2], 0, 1e-3)
        self.assertTrue(test_vals, str(np.exp(salt_kcl.log_gamma(param[:, 1]))) + " & " + str(param[:, 2]))

    def test_apparent_molal_enthalpy(self):
        # parameters in [temperature (C), molality, apparent relative molal enthalpy (kJ/mol)]
        param = np.array([[25, 1.0, -0.06],
                          [25, 0.1, 0.34],
                          [25, 0.5, 0.24],
                          [25, 0.3, 0.33],
                          [25, 2.0, -0.67],
                          [25, 3.0, -1.21],
                          [50, 0.5, 0.79],
                          [50, 3.0, 0.30],
                          [50, 0.1, 0.59],
                          [50, 0.3, 0.75]])
        # converting to [temperature (K), molality, apparent relative molal enthalpy (kJ/mol)]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        # testing params up to a precision of 10^-2
        salt_kcl = kcl.KClPropertiesPabalanPitzer(param[:, 0], (1 / un.atm_2_bar(1)))
        test_vals = np.allclose(salt_kcl.apparent_molal_enthalpy(param[:, 1]) / 1e3, param[:, 2], 0, 1e-2)
        self.assertTrue(test_vals, str(salt_kcl.apparent_molal_enthalpy(param[:, 1]) / 1e3) + " & " + str(param[:, 2]))


if __name__ == '__main__':
    unittest.main()
