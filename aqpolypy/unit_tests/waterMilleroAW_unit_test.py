"""
Unit Test for WaterPropertiesFineMillero
"""
import unittest
import aqpolypy.water.WaterPropertiesFineMillero as fm


class TestWaterMilleroAW(unittest.TestCase):

    # Testing Boltzman constant
    def test_density(self):
        # density of water at 25 C from Millero paper (1.012107 = specific volume)
        measured_density = (1 / 1.002961) * 1000
        # testing density
        wfm = fm.WaterPropertiesFineMillero(298.15)
        self.assertEqual(wfm.density(), measured_density)

    def test_molar_volume(self):
        pass

    def test_dielectric_constant(self):
        pass

    def test_compressibility(self):
        pass

if __name__ == '__main__':
    unittest.main()