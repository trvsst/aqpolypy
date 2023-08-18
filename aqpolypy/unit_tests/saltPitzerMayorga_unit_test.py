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
import aqpolypy.salt.SaltPiMa25 as slt

class TestFreeEnergy(unittest.TestCase):

    def setUp(self):
        # define dict
        self.temp = 298.15
        self.p_nacl = nacl.NaClPropertiesRogersPitzer(self.temp)
        num_pnts = 200
        self.m_l = np.linspace(1e-2, 5, num_pnts)

        cmp = 'NaCl'
        self.slt_class = slt.SaltPropertiesPitzerMayorga(cmp)

    def test_osmotic(self):
        coeff_osmotic = self.p_nacl.osmotic_coeff(self.m_l)
        comp_osmotic = self.slt_class.osmotic_coeff(self.m_l)

        diff1 = np.max(np.abs(comp_osmotic-coeff_osmotic))
        self.assertTrue(diff1 < 1e-6)

    def test_activity(self):
        coeff_activity = self.p_nacl.log_gamma(self.m_l)
        comp_activity = self.slt_class.log_gamma(self.m_l)

        diff1 = np.max(np.abs(coeff_activity-comp_activity))
        self.assertTrue(diff1 < 1e-6)

if __name__ == '__main__':
    unittest.main()
