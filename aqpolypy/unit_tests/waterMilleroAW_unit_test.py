"""
Unit Test for WaterPropertiesFineMillero
"""
import numpy as np
import unittest
import aqpolypy.water.WaterPropertiesFineMillero as fm


class TestWaterMilleroAW(unittest.TestCase):

    # Testing density (Fine Millero)
    def test_density(self):
        # density of water at 25 C (298.15 K) from Millero paper (1.002961 = specific volume)
        measured_density = (1 / 1.002961) * 1000
        # testing density up to a precision of 10^-6
        wfm = fm.WaterPropertiesFineMillero(298.15)
        np.allclose(wfm.density(), measured_density, 0, 1e-6)

    # Testing molar volume (Fine Millero)
    def test_molar_volume(self):
        # molar volume of water at 0 C (273.15); specific volume * molecular weight
        measured_molarVolume = (1.000160 / 1e6) * 18.01534
        # testing molar volume with specific volume = 1.000160 cm^3/g
        wfm = fm.WaterPropertiesFineMillero(273.15)
        np.allclose(wfm.molar_volume(), measured_molarVolume, 0, 1e-6)

    # Testing dielectric constant (Archer Wang)
    def test_dielectric_constant(self):
        pass

    # Testing compressibility (Fine Millero)
    def test_compressibility(self):
        pass

if __name__ == '__main__':
    unittest.main()