"""
:module: debye_huckel_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for DebyeHuckel

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""
import numpy as np
import unittest
import aqpolypy.water.WaterMilleroAW as waw
import aqpolypy.water.WaterMilleroBP as wbp
import aqpolypy.salts_theory.Bjerrum as bj
import aqpolypy.salts_theory.DebyeHuckel as dh


class TestDebyeHuckel(unittest.TestCase):

    def test_debye_length(self):
        # parameters in [temperature (K), concentration of ions (M), debye length]
        param = np.array([[273.15, 0.2, 9.743481417661739],
                          [298.15, 0.2, 9.612546583321587],
                          [300, 0.2, 9.601522416604853],
                          [298.15, 0.5, 6.079508263553147],
                          [300, 0.8, 4.8007612083024265],
                          [300, 1, 4.293931362203302]])

        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])

        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        obj_dh = dh.DebyeHuckel(obj_bj)
        obj_dh_2 = dh.DebyeHuckel(obj_bj_2)

        # testing params up to a precision of 10^-2
        test_1 = np.allclose(obj_dh.debye_length(param[:, 1]), param[:, 2], 0, 1e-2)
        test_2 = np.allclose(obj_dh_2.debye_length(param[:, 1]), param[:, 2], 0, 1e-2)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_free_energy_db_excess(self):
        # parameters in [temperature (K), concentration of ions (M), ion size (A), free energy excess]
        param = np.array([[273.15, 0.2, 2, -0.000796],
                          [298.15, 0.2, 2, -0.0008287],
                          [300, 0.2, 3, -0.002629],
                          [298.15, 0.5, 1, -0.000421],
                          [300, 0.8, 5, -0.068292],
                          [300, 1, 5, -0.090934]])

        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])

        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        obj_dh = dh.DebyeHuckel(obj_bj)
        obj_dh_2 = dh.DebyeHuckel(obj_bj_2)

        # testing params up to a precision of 10^-5
        test_1 = np.allclose(obj_dh.free_energy_db_excess(param[:, 1], param[:, 2]), param[:, 3], 0, 1e-5)
        test_2 = np.allclose(obj_dh_2.free_energy_db_excess(param[:, 1], param[:, 2]), param[:, 3], 0, 1e-5)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_pot_chem_db_excess(self):
        # parameters in [temperature (K), concentration of ions (M), ion size (A), potential chem excess]
        param = np.array([[273.15, 0.2, 2, -0.296315],
                          [298.15, 0.2, 2, -0.307876],
                          [300, 0.2, 3, -0.284365],
                          [298.15, 0.5, 1, -0.505009],
                          [300, 0.8, 5, -0.365628],
                          [300, 1, 5, -0.385567]])

        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])

        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        obj_dh = dh.DebyeHuckel(obj_bj)
        obj_dh_2 = dh.DebyeHuckel(obj_bj_2)

        # testing params up to a precision of 10^-3
        test_1 = np.allclose(obj_dh.pot_chem_db_excess(param[:, 1], param[:, 2]), param[:, 3], 0, 1e-3)
        test_2 = np.allclose(obj_dh_2.pot_chem_db_excess(param[:, 1], param[:, 2]), param[:, 3], 0, 1e-3)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_pressure_db_excess(self):
        # parameters in [temperature (K), concentration of ions (M), ion size (A), pressure excess]
        param = np.array([[273.15, 0.2, 2, 0.013466],
                          [298.15, 0.2, 2, 0.013616],
                          [300, 0.2, 3, 0.018677],
                          [298.15, 0.5, 1, 0.011194],
                          [300, 0.8, 5, 0.035652],
                          [300, 1, 5, 0.036521]])

        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])

        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        obj_dh = dh.DebyeHuckel(obj_bj)
        obj_dh_2 = dh.DebyeHuckel(obj_bj_2)

        # testing params up to a precision of 10^-5
        test_1 = np.allclose(obj_dh.pressure_db_excess(param[:, 1], param[:, 2]), param[:, 3], 0, 1e-5)
        test_2 = np.allclose(obj_dh_2.pressure_db_excess(param[:, 1], param[:, 2]), param[:, 3], 0, 1e-5)
        self.assertTrue(test_1)
        self.assertTrue(test_2)


if __name__ == '__main__':
    unittest.main()
