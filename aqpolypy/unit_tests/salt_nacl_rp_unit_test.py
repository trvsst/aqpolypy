"""
:module: salt_nacl_rp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for SaltNaClRP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""
import numpy as np
import unittest
import aqpolypy.salt.SaltNaClRP as nacl


class TestSaltNaClRP(unittest.TestCase):

    def test_h_fun(self):
        pass

    def test_h_fun_gamma(self):
        pass

    def test_p_fun_gamma(self):
        pass

    def test_params(self):
        # parameters in [temperature (C), pressure (Bar), [vp = None, bp, 2cp]]
        param = np.array([[10, 1, None, 1.956e-5, -2.25e-6],
                          [30, 1, None, 1.069e-5, -1.07e-6],
                          [90, 1, None, 2.577e-6, -6.02e-8],
                          [40, 400, None, 6.779e-6, -7.19e-7],
                          [50, 800, None, 4.436e-6, -4.71e-7]])
        # converting to [temperature (K), pressure (atm), [vp = None, bp, 2cp]]
        param[:, 0] = 273.15 + param[:, 0]
        param[:, 1] = param[:, 1] / 1.01325
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals1 = np.allclose(salt_nacl.params()[1], param[:, 3], 0, 1e-2)
        test_vals2 = np.allclose(2 * salt_nacl.params()[2], param[:, 4], 0, 1e-2)
        self.assertTrue(test_vals1)
        self.assertTrue(test_vals2)

    def test_stoichiometry_coeffs(self):
        # parameters in stoichiometry coefficient[nu = 2, nu_prod = 1, z_prod = 1, nz_prod_plus = 1]
        param = np.array([2, 1, 1, 1])
        salt_nacl = nacl.NaClPropertiesRogersPitzer(300)
        test_vals = np.allclose(salt_nacl.stoichiometry_coeffs(), param, 0, 1e-6)
        self.assertTrue(test_vals)

    def test_ionic_strength(self):
        pass

    def test_molar_vol_infinite_dilution(self):
        # parameters in [temperature (C), pressure (Bar), partial molal volume of solute cm^3/mol]
        param = np.array([[10, 1, 1.506e1],
                          [50, 1, 1.774e1],
                          [90, 1, 1.710e1],
                          [20, 400, 1.789e1],
                          [50, 600, 1.967e1]])
        # converting to [temperature (K), pressure (atm), partial molal volume of solute m^3/mol]
        param[:, 0] = 273.15 + param[:, 0]
        param[:, 1] = param[:, 1] / 1.01325
        param[:, 2] = param[:, 2] / 1000000
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_nacl.molar_vol_infinite_dilution(), param, 0, 1e-2)
        self.assertTrue(test_vals)

    def test_density_sol(self):
        # parameters in [temperature (C), pressure (Bar), specific volume of NaCl(aq) cm^3/g]
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
        # converting to [temperature (K), pressure (atm), specific volume of NaCl(aq) kg/m^3]
        param[:, 0] = 273.15 + param[:, 0]
        param[:, 1] = param[:, 1] / 1.01325
        param[:, 3] = (1 / param[:, 3]) * 1000
        # testing params up to a precision of 10^-2
        salt_nacl = nacl.NaClPropertiesRogersPitzer(param[:, 0], param[:, 1])
        test_vals = np.allclose(salt_nacl.density_sol(param[:, 2]), param[:, 3], 0, 1e-2)
        self.assertTrue(test_vals)

    def test_molar_vol(self):
        pass

    def test_osmotic_coeff(self):
        pass

    def test_log_gamma(self):
        pass


if __name__ == '__main__':
    unittest.main()
