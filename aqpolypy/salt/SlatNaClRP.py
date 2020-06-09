"""
:module: SaltNaClRP
:platform: Unix, Windows, OS
:synopsis: Derived salt properties class utilizing Rogers Pitzer calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

import numpy as np
import aqpolypy.units.units as un
import aqpolypy.salt.SaltPropertiesABC as sp


class SaltPropertiesRogersPitzer(sp.SaltProperties):
    """
    Slat Properties

    """

    def __init__(self):
        """
        constructor

        :param :
        :param :
        :instantiate:

        """
        super().__init__()

        # Calculations
