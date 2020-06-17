"""
:module: polymer_peo_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for PEOSimple and PolymerPropertiesABC

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added tests for all methods in SaltPropertiesPitzer
"""
import unittest
import aqpolypy.polymer.PEOSimple as peo


class TestPolymerPEO(unittest.TestCase):

    def test_values(self):
        # tests the different values in the function

        # let us take the PEO with molecular weight 5000
        pl_peo = peo.PEOSimple(5000)

        self.assertEqual(pl_peo.k_length, 7.24)
        self.assertEqual(pl_peo.ratio_real_kuhn, 0.5027624309392266)
        self.assertEqual(pl_peo.p_name, pl_peo.name())
        self.assertEqual(pl_peo.melt_density, 1124.1152125306096)
        self.assertEqual(pl_peo.n_p, 57.06465437882804)
        self.assertEqual(pl_peo.n, 113.50222464360301)
        self.assertEqual(pl_peo.max_length, 413.148097702715)


if __name__ == '__main__':
    unittest.main()
