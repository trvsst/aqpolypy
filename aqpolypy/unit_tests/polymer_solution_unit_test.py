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
        f_v = 1/3.0
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
            polymer_sol = Ps.PolymerSolution(phi, x, p, n_kuhn, f_v, chi, df_w/te, df_p/te)
            x_sol[ind] = polymer_sol.x
            p_sol[ind] = polymer_sol.y

        self.assertTrue(np.allclose(x_sol, x_comp, rtol=0.0, atol=1e-7))
        self.assertTrue(np.allclose(p_sol, p_comp, rtol=0.0, atol=1e-7))

    def test_free(self):
        """ checks free energy, chemical potential and osmotic pressure
        """

        num_pnts = 10

        f_comp = np.array([-7.32873495, -6.70644881, -6.08600374, -5.46708749, -4.84935289, -4.23238319, -3.61561235,
                           -2.99817553, -2.3786123, -1.75418721])

        m_comp = np.array([8.01386596, 7.98835116, 7.96662221, 7.94910351, 7.93637711, 7.92974097, 7.93187588,
                           7.94807369, 7.98908031, 8.07916751])

        n_kuhn = 100
        f_v = 1/3.0
        chi = 0.5
        de_w = 1800
        delta_w = np.pi / 5
        de_p = 2000
        delta_p = np.pi / 8

        def sf(z):
            return -np.log(0.5 * (1 - np.cos(z)))

        te = 300
        df_w = de_w - te * sf(delta_w)
        df_p = de_p - te * sf(delta_p)

        x_ini = 0.7
        p_ini = 0.4

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)
        osm = np.zeros_like(phi_val)
        chem = np.zeros_like(phi_val)
        x = x_ini
        p = p_ini
        for ind, phi in enumerate(phi_val):
            polymer_sol = Ps.PolymerSolution(phi, x, p, n_kuhn, f_v, chi, df_w / te, df_p / te)
            x = polymer_sol.x
            p = polymer_sol.y
            free[ind] = polymer_sol.free()
            chem[ind] = polymer_sol.chem_potential()
            osm[ind] = polymer_sol.osm_pressure()

        # exact relation between osmotic pressure, free energy, chemical potential and polymer density phi
        rel = osm+free-phi_val*chem

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-7))
        self.assertTrue(np.allclose(chem, m_comp, rtol=0.0, atol=1e-7))
        self.assertTrue(np.amax(np.abs(rel)) < 1e-11)


if __name__ == '__main__':
    unittest.main()
