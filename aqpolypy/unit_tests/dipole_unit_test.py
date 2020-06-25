"""
:module: dipole_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for Dipole

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
import aqpolypy.salts_theory.Dipole as dp


class TestDipole(unittest.TestCase):

    def test_free_energy_dp_excess(self):
        # parameters in [temperature (K), concentration of ions (M), ion size (A), x_pair, free energy excess]
        param = np.array([[273.15, 1.0, 8, 0.4, -0.0036],
                          [298.15, 1.0, 8, 0.4, -0.0038],
                          [300.00, 1.0, 9, 0.6, -0.0064],
                          [298.15, 2.0, 7, 0.4, -0.0074],
                          [300.00, 4.0, 5, 0.5, -0.0086],
                          [300.00, 3.5, 5, 0.7, -0.0071]])

        obj_water_bp = wbp.WaterPropertiesFineMillero(param[:, 0])
        obj_water_aw = waw.WaterPropertiesFineMillero(param[:, 0])

        obj_bj = bj.Bjerrum(obj_water_bp)
        obj_bj_2 = bj.Bjerrum(obj_water_aw)

        obj_dp = dp.Dipole(obj_bj)
        obj_dp_2 = dp.Dipole(obj_bj_2)

        # testing params up to a precision of 10^-4
        test_1 = np.allclose(obj_dp.free_energy_dp_excess(param[:, 1], param[:, 2], param[:, 3]), param[:, 4], 0, 1e-4)
        test_2 = np.allclose(obj_dp_2.free_energy_dp_excess(param[:, 1], param[:, 2], param[:, 3]), param[:, 4], 0, 1e-4)
        self.assertTrue(test_1)
        self.assertTrue(test_2)

    def test_pot_chem_dp_excess(self):
        # parameters in [temperature (K), concentration of ions (M), ion size (A), x_pair, [mu_p, mu_p, mu_0]]
        param = np.array([[273.15, 1.0, 8, 0.4, -1.39225705e-05, -1.39225705e-05, -5.82590279e-02],
                          [298.15, 1.0, 8, 0.4, -1.47196945e-05, -1.47196945e-05, -6.08761754e-02],
                          [300.00, 1.0, 9, 0.6, -2.21908764e-05, -2.21908764e-05, -4.87829304e-02],
                          [298.15, 2.0, 7, 0.4, -4.60404423e-05, -4.60404423e-05, -8.97120601e-02],
                          [300.00, 4.0, 5, 0.5, -3.07170181e-05, -3.07170181e-05, -1.14731930e-01],
                          [300.00, 3.5, 5, 0.7, -9.91702493e-06, -9.91702493e-06, -7.65120554e-02]])

        # testing params [mu_p, mu_p, mu_0]
        for i in range(len(param)):
            for j in range(3):
                obj_water_bp = wbp.WaterPropertiesFineMillero(param[i, 0])
                obj_water_aw = waw.WaterPropertiesFineMillero(param[i, 0])

                obj_bj = bj.Bjerrum(obj_water_bp)
                obj_bj_2 = bj.Bjerrum(obj_water_aw)

                obj_dp = dp.Dipole(obj_bj)
                obj_dp_2 = dp.Dipole(obj_bj_2)

                dp_method = obj_dp.pot_chem_dp_excess(param[i, 1], param[i, 2], param[i, 3])
                dp_method_2 = obj_dp_2.pot_chem_dp_excess(param[i, 1], param[i, 2], param[i, 3])

                # testing params [mu_p, mu_p] up to a precision of 10^-7
                test_1 = np.allclose(dp_method[j], param[i, 4 + j], 0, 1e-7)
                test_2 = np.allclose(dp_method_2[j], param[i, 4 + j], 0, 1e-7)

                # testing params [mu_0] up to a precision of 10^-4
                if j == 2:
                    test_1 = np.allclose(dp_method[j], param[i, 4 + j], 0, 1e-4)
                    test_2 = np.allclose(dp_method_2[j], param[i, 4 + j], 0, 1e-4)

                self.assertTrue(test_1)
                self.assertTrue(test_2)


if __name__ == '__main__':
    unittest.main()
