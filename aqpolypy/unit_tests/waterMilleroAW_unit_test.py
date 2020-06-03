"""
Unit Test for WaterPropertiesFineMillero
"""
import numpy as np
import unittest
import aqpolypy.water.WaterPropertiesFineMillero as fm


class TestWaterMilleroAW(unittest.TestCase):

    # Testing density (Fine Millero)
    def test_density(self):
        # density of water at [temperature(C), pressure(bar), specific volume] Millero TABLEIV
        param = np.array([[5, 1, 1.000036],
                          [30, 1, 1.004369],
                          [75, 1, 1.025805],
                          [35, 100, 1.001597],
                          [55, 300, 1.001642]])
        # converting param to [temperature(K), pressure(atm), density]
        param[:, 0] = 273.15 + param[:, 0]
        param[:, 1] = param[:, 1] / 1.01325
        param[:, 2] = (1 / param[:, 2]) * 1000
        # testing density up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.density(), param[:, 2], 0, 1e-6)
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
        param[:, 0] = 273.15 + param[:, 0]
        param[:, 1] = param[:, 1] / 1.01325
        param[:, 2] = (param[:, 2] / 1e6) * fm.WaterPropertiesFineMillero(0).MolecularWeight
        # testing molar volume up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.molar_volume(), param[:, 2], 0, 1e-6)
        self.assertTrue(test_vals)

    # Testing dielectric constant (Archer Wang)
    def test_dielectric_constant(self):
        # dielectric constant of water at [temperature(K), pressure(MPa), dielectric constant] Archer & Wang Table 4
        param = np.array([[293.15, 0.1, 80.20],
                          [318.15, 0.1, 71.50],
                          [353.15, 0.1, 60.87],
                          [273.15, 1, 87.94],
                          [283.15, 10, 84.33]])
        # converting param to [temperature(K), pressure(atm), dielectric constant]
        param[:, 1] = param[:, 1] / 0.101325
        # testing dielectric constant up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.dielectric_constant(), param[:, 2], 0, 1e-6)
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
        param[:, 0] = 273.15 + param[:, 0]
        param[:, 1] = param[:, 1] / 1.01325
        param[:, 2] = (param[:, 2] * 1e6) / 1.01325
        # testing compressibility up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(param[:, 0], param[:, 1])
        test_vals = np.allclose(wfm.compressibility(), param[:, 2], 0, 1e-6)
        self.assertTrue(test_vals)


if __name__ == '__main__':
    unittest.main()
