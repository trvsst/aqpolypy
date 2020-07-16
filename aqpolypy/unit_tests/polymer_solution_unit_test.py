"""
:module: polymer_solution_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for the free energy calculation

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, July2020
.. history:
"""
import unittest
import numpy as np

import aqpolypy.free_energy_polymer.PolymerSolution as Ps


class TestPolymerFreeSpherical(unittest.TestCase):

    def test_hb_frac(self):
        """checks that the fraction of hydrogen bonds is correct"""

        temp_ini = 273.15
        temp_fin = 400 + temp_ini
        num_steps = 10
        temp = np.linspace(temp_ini, temp_fin, num_steps)

        x_comp = np.array([0.69187745, 0.63804776, 0.57778352, 0.51745377, 0.46128402, 0.41119443, 0.36761076,
                           0.33017857, 0.29821508, 0.27095495])
        p_comp = np.array([0.73038565, 0.70204064, 0.66398896, 0.62191529, 0.57960759, 0.53919457, 0.50172266,
                           0.46758977, 0.43682477, 0.40925976])

        phi = 0.5
        n_kuhn = 100
        f_v = 3.0
        chi = 0.5
        de_w = 1800
        delta_w = np.pi / 5
        de_p = 2000
        delta_p = np.pi/8

        x_sol = np.zeros_like(temp)
        p_sol = np.zeros_like(temp)

        def sf(z):
            return -np.log(0.5*(1-np.cos(z)))

        solns = np.array([0.7, 0.7])
        for ind, te in enumerate(temp):
            df_w = de_w-te*sf(delta_w)
            df_p = de_p-te*sf(delta_p)
            x = solns[0]
            p = solns[1]
            polymer_sol = Ps.PolymerSolution(phi, n_kuhn, f_v, chi, df_w/te, df_p/te)
            solns = polymer_sol.solv_eqns(x, p)
            x_sol[ind] = solns[0]
            p_sol[ind] = solns[1]

        self.assertTrue(np.allclose(x_sol, x_comp, rtol=0.0, atol=1e-7))
        self.assertTrue(np.allclose(p_sol, p_comp, rtol=0.0, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
