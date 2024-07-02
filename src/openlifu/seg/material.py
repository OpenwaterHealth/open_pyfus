from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

@dataclass
class Material:
    id: str = "material"
    name: str = "Material"
    sound_speed: float = 1500.0 # m/s
    density: float = 1000.0 # kg/m^3
    attenuation: float = 0.0 # dB/cm/MHz
    specific_heat: float = 4182.0 # J/kg/K
    thermal_conductivity: float = 0.598 # W/m/K
    param_ids: Tuple[str] = field(default_factory= lambda: ("sound_speed", "density", "attenuation", "specific_heat", "thermal_conductivity"), init=False, repr=False)

    @classmethod
    def param_info(cls, param_id: str):
        INFO = {"sound_speed":{"id":"sound_speed",
                               "name": "Speed of Sound",
                               "units": "m/s"},
                "density":{"id":"density",
                           "name": "Density",
                           "units": "kg/m^3"},
                "attenuation":{"id":"attenuation",
                               "name": "Attenuation",
                               "units": "dB/cm/MHz"},
                "specific_heat":{"id":"specific_heat",
                                 "name": "Specific Heat",
                                 "units": "J/kg/K"},
                "thermal_conductivity":{"id":"thermal_conductivity",
                                        "name": "Thermal Conductivity",
                                        "units": "W/m/K"}}
        if param_id not in INFO.keys():
            raise ValueError(f"Parameter {param_id} not found.")
        return INFO[param_id]

    def get_param(self, param_id: str):
        if param_id not in self.param_ids:
            raise ValueError(f"Parameter {param_id} not found.")
        return self.__getattribute__(param_id)

    @staticmethod
    def get_materials(material_id="all", as_dict=True):
        material_id = ("water", "tissue", "skull", "air", "standoff") if material_id == "all" else material_id
        if isinstance(material_id, (list, tuple)):
            materials = {m: Material.get_materials(m, as_dict=False) for m in material_id}
        elif material_id in MATERIALS:
            materials = MATERIALS[material_id]
        else:
            raise ValueError(f"Material {material_id} not found.")
        if as_dict:
            return {materials.id: materials}
        else:
            return materials

    @staticmethod
    def from_dict(d):
        if isinstance(d, (list, tuple)):
            return {dd['id']: Material.from_dict(dd) for dd in d}
        elif isinstance(d, str):
            return Material.get_materials(d, as_dict=False)
        elif isinstance(d, Material):
            return d
        else:
            return Material(**d)


WATER = Material(id="water",
                 name="water",
                 sound_speed=1500.0,
                 density=1000.0,
                 attenuation=0.0,
                 specific_heat=4182.0,
                 thermal_conductivity=0.598)

TISSUE = Material(id="tissue",
                  name="tissue",
                  sound_speed=1540.0,
                  density=1000.0,
                  attenuation=0.0,
                  specific_heat=3600.0,
                  thermal_conductivity=0.5)

SKULL = Material(id="skull",
                 name="skull",
                 sound_speed=4080.0,
                 density=1900.0,
                 attenuation=0.0,
                 specific_heat=1100.0,
                 thermal_conductivity=0.3)

AIR = Material(id="air",
               name="air",
               sound_speed=344.0,
               density=1.25,
               attenuation=0.0,
               specific_heat=1012.0,
               thermal_conductivity=0.025)

STANDOFF = Material(id="standoff",
                    name="standoff",
                    sound_speed=1420.0,
                    density=1000.0,
                    attenuation=1.0,
                    specific_heat=4182.0,
                    thermal_conductivity=0.598)

MATERIALS = {"water": WATER,
             "tissue": TISSUE,
             "skull": SKULL,
             "air": AIR,
             "standoff": STANDOFF}
