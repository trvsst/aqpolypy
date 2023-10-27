"""
:module: electrolyte_solution_virial_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for electrolyte solution with implict model

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, October2023
.. history:
..
"""
import numpy as np
import unittest

import aqpolypy.free_energy_polymer.ElectrolyteSolutionImplicitVirial as Ev


class TestFreeEnergy(unittest.TestCase):

    def setUp(self):
        # define dict
        self.temp = 298.15

        self.m_l = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])

        b_param = 1.4
        e_s = np.array([0.0716, 0.0058])
        self.im_model = Ev.ElectrolyteSolutionVirial(self.m_l, self.temp, b_param, e_s)

    def test_osmotic(self):
        coeff_osmotic = np.array([0.92436877, 0.92197449, 0.92213676, 0.92368136, 0.92608443, 0.9290711, 0.93248152,
                                  0.93621584, 0.94020828, 0.94880009, 0.95802855, 0.96776878, 0.97794727, 0.98851844,
                                  1.01649422, 1.04651144, 1.07846821, 1.11232135, 1.14805283, 1.18565618, 1.22513046,
                                  1.26647742])

        comp_osmotic = self.im_model.osmotic_coeff()

        diff1 = np.max(np.abs(comp_osmotic-coeff_osmotic))
        self.assertTrue(diff1 < 0.002)

    def test_activity(self):
        coeff_activity = ([0.73460657, 0.71031024, 0.69464229, 0.68383267, 0.67614239, 0.67062776, 0.6667212,
                          0.66405605, 0.66238254, 0.66134933, 0.66268778, 0.66585392, 0.67050834, 0.67642909,
                          0.69590087, 0.72115566, 0.75171468, 0.78751067, 0.82872017, 0.87569213, 0.92891775,
                          0.98902066])
        comp_activity = np.exp(self.im_model.log_gamma())

        diff1 = np.max(np.abs(coeff_activity-comp_activity))
        self.assertTrue(diff1 < 0.004)

if __name__ == '__main__':
    unittest.main()
