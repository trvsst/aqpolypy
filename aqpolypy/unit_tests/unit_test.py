"""
Unit Test for Units file
"""
import unittest
import aqpolypy.units.Units as un


class TestUnits(unittest.TestCase):

    # Testing Boltzman constant
    def test_k_boltzmann(self):
        self.assertEqual(un.k_boltzmann(), 1.380649e-23)

    # Testing avogadro
    def test_avogadro(self):
        pass

    # Testing r gas
    def test_r_gas(self):
        pass

    # Testing atm to pascal & atm to bar
    def test_atm_conversion(self):
        pass

    # Testing celsius to kelvin conversion
    def test_kelvin_conversion(self):
        pass

    # Testing meter to angstrom conversion
    def test_meter_conversion(self):
        pass

    # Testing mol/litre to molecule/A^3 conversion
    def test_mol_lit_conversion(self):
        pass

    # Testing 1/4pieps0, e charge, e charge squared
    def test_electrostatic_constants(self):
        pass

