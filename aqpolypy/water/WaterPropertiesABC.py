from abc import ABC, abstractmethod


class WaterProperties(ABC):
    def __init__(self):
        self.MolecularWeight = 18.01534

    @abstractmethod
    def density(self):
        pass

    def molar_volume(self):
        pass

    def dielectric_constant(self):
        pass

    def compressibility(self):
        pass
