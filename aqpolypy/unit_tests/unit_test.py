"""
Testing functions from Units file
"""
import aqpolypy.units.Units as un

# Testing Boltzman constant
print("TESTING for Boltzman constant")
print("Boltzman constant is " + str(un.k_boltzmann()))

# Testing avogadro
print("\nTESTING for avogadro number")
print("Avogadro's number is " + str(un.avogadro()))

# Testing r gas
print("\nTESTING for Gas constant ")
print("Gas constant r is " + str(un.r_gas()))

# Testing atm to pascal & atm to bar
print("\nTESTING atm conversions")
p_atm = 2.3
p_pa = un.atm_2_pascal(p_atm)
p_bar = un.atm_2_bar(p_atm)
print(str(p_atm) + " atm is " + str(p_pa) + " Pa, " + str(p_bar) + " bar")

# Testing celsius to kelvin conversion
print("\nTESTING C to K conversion")
t_C = 27.84
t_K = un.celsius_2_kelvin(t_C)
print(str(t_C) + " Celsius is " + str(t_K) + " Kelvin")

# Testing meter to angstrom conversion
print("\nTESTING meter to angstrom conversion")
m = 4.487
A = un.m_2_angstrom(m)
print(str(m) + " meter is " + str(A) + " angstrom")

# Testing mol/litre to molecule/A^3 conversion
print("\nTESTING mol/L to molecule/A^3 conversion")
Molar = 6.87
mol_per_A3 = un.mol_lit_2_mol_angstrom(Molar)
print(str(Molar) + " mol/L is " + str(mol_per_A3) + " molecule/A3")

# Testing 1/4pieps0, e charge, e charge squared
print("\nTESTING 1/4pieps0, e charge, e charge squared")
print("one over 4*PI*epsilon0 is " + str(un.one_over4pi_epsilon0()))
print("e charge is " + str(un.e_charge()))
print("e charge squared is " + str(un.e_square()))


