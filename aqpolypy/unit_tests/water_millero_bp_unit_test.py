"""
:module: water_millero_bp_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for WaterMilleroBP

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""
import numpy as np
import unittest
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as fm


class TestWaterMilleroBP(unittest.TestCase):

    # Testing density (Fine Millero)
    def test_density(self):
        # density of water at [temperature(C), pressure(bar), specific volume] Millero TABLEIV
        param = np.array([[5, 1, 1.000036],
                          [30, 1, 1.004369],
                          [75, 1, 1.025805],
                          [30, 100, 0.999939],
                          [55, 300, 1.001642]])
        # converting param to [temperature(K), pressure(atm), specific volume]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        # testing density up to a precision of 10^-4
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(1e3 / wfm.density(), param[:, 2], 0, 1e-4)
        self.assertTrue(test_vals)

    # Testing molar volume (Fine Millero)
    def test_molar_volume(self):
        # molar volume of water at [temperature(C), pressure(bar), specific volume] Millero TABLE IV
        param = np.array([[5, 1, 1.000036],
                          [30, 1, 1.004369],
                          [75, 1, 1.025805],
                          [35, 100, 1.001597],
                          [55, 300, 1.001642]])
        # converting param to [temperature(K), pressure(atm), molar volume]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 2] = (param[:, 2] / 1e6) * fm.WaterPropertiesFineMillero(300).MolecularWeight
        # testing molar volume up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.molar_volume(), param[:, 2], 0, 1e-6)
        self.assertTrue(test_vals)

    # Testing dielectric constant (Bradley Pitzer)
    def test_dielectric_constant(self):
        # dielectric constant of water at [temperature(K), pressure(MPa), dielectric constant] Archer & Wang Table 4
        param = np.array([[338.15, 0.1, 65.22],
                          [318.15, 0.1, 71.50],
                          [353.15, 0.1, 60.87],
                          [273.15, 1, 87.94],
                          [278.15, 60, 88.20]])
        # converting param to [temperature(K), pressure(atm), dielectric constant]
        param[:, 1] = param[:, 1] * 1e6 / un.atm_2_pascal(1)
        # testing dielectric constant up to a precision of 10^-1
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.dielectric_constant(), param[:, 2], 0, 1e-1)
        self.assertTrue(test_vals)

    # Testing compressibility (Fine Millero)
    def test_compressibility(self):
        # compressibility of water at [temperature(C), pressure(bar), compressibility (10^6 bar-1)] Millero TABLE V
        param = np.array([[5, 1, 49.175],
                          [30, 1, 44.771],
                          [75, 1, 45.622],
                          [35, 100, 43.305],
                          [55, 300, 40.911]])
        # converting param to [temperature(K), pressure(atm), compressibility (atm-1)]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        param[:, 2] = (param[:, 2] / 1e6) * un.atm_2_bar(1)
        # testing compressibility up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.compressibility(), param[:, 2], 0, 1e-6)
        self.assertTrue(test_vals)

    # Testing osmotic coefficient (Bradley Pitzer)
    def test_a_phi(self):
        # osmotic coefficient of water [temperature(C), Pressure(bar), osmotic coefficient] Bradley & Pitzer TABLE II
        param = np.array([[10, 100, 3.80e-1],
                          [60, 100, 4.17e-1],
                          [120, 100, 4.82e-1],
                          [70, 400, 4.19e-1],
                          [25, 600, 3.81e-1]])
        # converting param to [temperature(K), pressure(atm), osmotic coefficient]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        # testing osmotic coefficient up to a precision of 10^-3
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.a_phi(), param[:, 2], 0, 1e-3)
        self.assertTrue(test_vals)

    # Testing apparent molal volume (Bradley Pitzer)
    def test_a_v(self):
        # apparent molal volume [temperature(C), Pressure(bar), molal vol (cc/mol)] Bradley & Pitzer TABLE IV
        param = np.array([[10, 100, 1.61],
                          [60, 100, 2.55],
                          [120, 100, 4.91],
                          [70, 400, 2.59],
                          [25, 600, 1.66]])
        # converting param to [temperature(K), pressure(atm), molal vol (cc/mol)]
        param[:, 0] = un.celsius_2_kelvin(param[:, 0])
        param[:, 1] = param[:, 1] / un.atm_2_bar(1)
        # testing apparent molal volume up to a precision of 10^-2
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.a_v(), param[:, 2], 0, 1e-2)
        self.assertTrue(test_vals)


if __name__ == '__main__':
    unittest.main()
