"""
:module: bjerrum_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for Bjerrum

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


class TestBjerrum(unittest.TestCase):

    def test_bjerrum_length(self):
        # parameters to test (water objects at  param (K) temperatures)
        param = np.array([273.15, 298.15, 300, 310, 320])
        obj_water_bp = wbp.WaterPropertiesFineMillero(param)
        obj_water_aw = waw.WaterPropertiesFineMillero(param)

        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        # testing params up to a precision of 10^-2
        bj_length_param = np.array([6.95955102, 7.15043775, 7.16686699, 7.26135158, 7.36525138])
        test_1 = np.allclose(obj_bj.bjerrum_length, bj_length_param, 0, 1e-2)
        test_2 = np.allclose(obj_bj_2.bjerrum_length, bj_length_param, 0, 1e-2)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_temp_star(self):
        # parameters in [temperature (K), radius (A), reduced temperature]
        param = np.array([[273.15, 2, 0.28737486],
                          [298.15, 2, 0.27970316],
                          [300, 2, 0.27906197],
                          [298.15, 5, 0.69925789],
                          [300, 0.5, 0.06976549],
                          [300, 1, 0.13953098]])
        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])
        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        # testing params up to a precision of 10^-3
        test_1 = np.allclose(obj_bj.temp_star(param[:, 1]), param[:, 2], 0, 1e-3)
        test_2 = np.allclose(obj_bj_2.temp_star(param[:, 1]), param[:, 2], 0, 1e-3)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_bjerrum_constant_approx(self):
        # parameters in [temperature (K), radius (A), Bjerrum constant (approx)]
        param = np.array([[273.15, 2, 175.6241625002495],
                          [298.15, 2, 197.37695820415016],
                          [300, 2, 199.34627081662805],
                          [298.15, 5, 4.437998539403452],
                          [300, 1, 5670.345869397299]])

        # testing params up to a precision of 10^2
        for i in range(len(param)):
            obj_water_bp = wbp.WaterPropertiesFineMillero(param[i, 0])
            obj_water_aw = waw.WaterPropertiesFineMillero(param[i, 0])
            obj_bj = bj.Bjerrum(obj_water_bp).bjerrum_constant_approx(param[i, 1])
            obj_bj_2 = bj.Bjerrum(obj_water_aw).bjerrum_constant_approx(param[i, 1])
            test_1 = np.allclose(obj_bj[0], param[i, 2], 0, 1e2)
            test_2 = np.allclose(obj_bj_2[0], param[i, 2], 0, 1e2)
            self.assertTrue(test_1)
            self.assertTrue(test_2)

    def test_bjerrum_constant(self):
        # parameters in [temperature (K), radius (A), Bjerrum constant]
        param = np.array([[273.15, 2, 177.15962555],
                          [298.15, 2, 199.08793975],
                          [300, 2, 201.07303174],
                          [298.15, 5, 4.48179914],
                          [300, 1, 5697.97522674]])

        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])
        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        # testing params up to a precision of 10
        test_1 = np.allclose(obj_bj.bjerrum_constant(param[:, 1]), param[:, 2], 0, 1e1)
        test_2 = np.allclose(obj_bj_2.bjerrum_constant(param[:, 1]), param[:, 2], 0, 1e1)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_b_parameter(self):

        temp = 298.15

        b_calc = 1/3.0443153743297127

        obj_water_bp = wbp.WaterPropertiesFineMillero(temp)
        obj_bj = bj.Bjerrum(obj_water_bp)

        b_param = obj_bj.b_parameter()

        self.assertTrue(np.abs(b_calc-b_param)< 1e-8)

if __name__ == '__main__':
    unittest.main()
