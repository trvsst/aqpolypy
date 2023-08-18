"""
:module: electrolyte_solution_implicit_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for electrolyte solution with implict model

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2023
.. history:
..
"""
import numpy as np
import unittest

import aqpolypy.units.units as un
import aqpolypy.salt.SaltNaClRP as nacl
import aqpolypy.free_energy_polymer.ElectrolyteSolutionImplicitSolvent as Is


class TestFreeEnergy(unittest.TestCase):

    def setUp(self):
        # define dict
        self.temp = 298.15
        self.p_nacl = nacl.NaClPropertiesRogersPitzer(self.temp)
        num_pnts = 200
        self.m_l = np.linspace(1e-2, 5, num_pnts)
        b_par = 1.39401205
        k_b = 0.37006548
        e_s = np.array([0.4573532, 1.08935634])
        self.im_model = Is.ElectrolyteSolutionImplicit(self.m_l, self.temp, b_par, k_b, e_s)
        self.im_model.solve_bjerrum_equation()

    def test_osmotic(self):
        coeff_osmotic = self.p_nacl.osmotic_coeff(self.m_l)
        comp_osmotic = self.im_model.osmotic_coeff()

        diff1 = np.max(np.abs(comp_osmotic-coeff_osmotic))
        self.assertTrue(diff1 < 0.002)

    def test_activity(self):
        coeff_activity = self.p_nacl.log_gamma(self.m_l)
        comp_activity = self.im_model.log_gamma()

        diff1 = np.max(np.abs(coeff_activity-comp_activity))
        self.assertTrue(diff1 < 0.004)

if __name__ == '__main__':
    unittest.main()
