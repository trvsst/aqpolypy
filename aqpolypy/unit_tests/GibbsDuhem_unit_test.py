"""
:module: GibbsDuhem_unit_test
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
import aqpolypy.free_energy_polymer.GibbsDuhem as gd

class TestFreeEnergy(unittest.TestCase):

    def setUp(self):
        # define dict
        temp = 298.15
        p_nacl = nacl.NaClPropertiesRogersPitzer(temp)

        num_pnts = 1000
        self.m_l = np.linspace(1e-6, 5, num_pnts)
        self.coeff_osmotic = p_nacl.osmotic_coeff(self.m_l)

        self.coeff_activity = p_nacl.log_gamma(self.m_l)
        self.gibbs_duhem = gd.GibbsDuhem(self.m_l, self.coeff_osmotic, self.coeff_activity, temp)

    def test_calculated_log_gamma(self):

        diff1 = self.gibbs_duhem.calculate_log_gamma()-self.coeff_activity
        self.assertTrue(np.max(np.abs(diff1)) < 0.004)

    def test_calculated_osmotic(self):

        diff1 = self.gibbs_duhem.calculate_osmotic_coeff() - self.coeff_osmotic
        self.assertTrue(np.max(np.abs(diff1)) < 0.007)

if __name__ == '__main__':
    unittest.main()
