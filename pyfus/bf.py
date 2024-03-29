"""
Beamforming classes and functions.

This module contains the classes and functions for beamforming.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pyfus import geo, xdc
from typing import Any, Tuple, Optional, ClassVar
from pyfus.util.units import getunitconversion
from pyfus.xdc import Transducer
import xarray as xa
import logging

@dataclass
class Pulse:
    """
    Class for representing a sinusoidal pulse
    
    :ivar frequency: Frequency of the pulse in Hz
    :ivar amplitude: Amplitude of the pulse in Pa
    :ivar duration: Duration of the pulse in s
    """

    frequency: float = 1.0 # Hz
    amplitude: float = 1.0 # Pa
    duration: float = 1.0 # s

    def calc_pulse(self, t: np.array):
        """
        Calculate the pulse at the given times
        
        :param t: Array of times to calculate the pulse at (s)
        :returns: Array of pulse values at the given times
        """
        return self.amplitude * np.sin(2*np.pi*self.frequency*t)
    
    def calc_time(self, dt: float):
        """
        Calculate the time array for the pulse for a particular timestep

        :param dt: Time step (s)
        :returns: Array of times for the pulse (s)
        """
        return np.arange(0, self.duration, dt)
    
    def get_table(self):
        """
        Get a table of the pulse parameters
        
        :returns: Pandas DataFrame of the pulse parameters
        """
        records = [{"Name": "Frequency", "Value": self.frequency, "Unit": "Hz"},
                   {"Name": "Amplitude", "Value": self.amplitude, "Unit": "Pa"},
                   {"Name": "Duration", "Value": self.duration, "Unit": "s"}]
        return pd.DataFrame.from_records(records)
    
    def to_dict(self):
        """
        Convert the pulse to a dictionary

        :returns: Dictionary of the pulse parameters
        """
        return {"frequency": self.frequency,
                "amplitude": self.amplitude,
                "duration": self.duration,
                "class": "Pulse"}
    
    @staticmethod
    def from_dict(d):
        """
        Create a pulse from a dictionary

        :param d: Dictionary of the pulse parameters
        :returns: Pulse object
        """
        return Pulse(frequency=d["frequency"], amplitude=d["amplitude"], duration=d["duration"])
    
@dataclass
class Sequence:
    """
    Class for representing a sequence of pulses

    :ivar pulse_interval: Interval between pulses in the sequence (s)
    :ivar pulse_count: Number of pulses in the sequence
    :ivar pulse_train_interval: Interval between pulse trains in the sequence (s)
    :ivar pulse_train_count: Number of pulse trains in the sequence
    """
    pulse_interval: float = 1.0 # s
    pulse_count: int = 1
    pulse_train_interval: float = 1.0 # s
    pulse_train_count: int = 1

    def get_table(self):
        """
        Get a table of the sequence parameters

        :returns: Pandas DataFrame of the sequence parameters
        """
        records = [{"Name": "Pulse Interval", "Value": self.pulse_interval, "Unit": "s"},
                   {"Name": "Pulse Count", "Value": self.pulse_count, "Unit": ""},
                   {"Name": "Pulse Train Interval", "Value": self.pulse_train_interval, "Unit": "s"},
                   {"Name": "Pulse Train Count", "Value": self.pulse_train_count, "Unit": ""}]
        return pd.DataFrame.from_records(records)
    
    @staticmethod
    def from_dict(d):
        """
        Create a sequence from a dictionary

        :param d: Dictionary of the sequence parameters
        :returns: Sequence object
        """
        return Sequence(pulse_interval=d["pulse_interval"], 
                        pulse_count=d["pulse_count"],
                        pulse_train_interval=d["pulse_train_interval"], 
                        pulse_train_count=d["pulse_train_count"])
    
@dataclass
class FocalPattern(ABC):
    """
    Abstract base class for representing a focal pattern

    :ivar target_pressure: Target pressure of the focal pattern in Pa
    """
    target_pressure: float = 1.0 # Pa

    @abstractmethod
    def get_targets(self, target: geo.Point):
        """
        Get the targets of the focal pattern
        
        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        pass

    @abstractmethod
    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci
        """
        pass

    def to_dict(self):
        """
        Convert the focal pattern to a dictionary

        :returns: Dictionary of the focal pattern parameters
        """
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d):
        """
        Create a focal pattern from a dictionary
        
        :param d: Dictionary of the focal pattern parameters
        :returns: FocalPattern object
        """
        d = d.copy()
        class_constructor = globals()[d.pop("class")]
        return class_constructor(**d)

@dataclass
class SingleFocus(FocalPattern):
    """
    Class for representing a single focus

    :ivar target_pressure: Target pressure of the focal pattern in Pa
    """
    def get_targets(self, target: geo.Point):
        """
        Get the targets of the focal pattern
        
        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        return [target.copy()]

    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci (1)
        """
        return 1
    
@dataclass
class WheelPattern(FocalPattern):
    """
    Class for representing a wheel pattern

    :ivar target_pressure: Target pressure of the focal pattern in Pa
    :ivar center: Whether to include the center point of the wheel pattern
    :ivar num_spokes: Number of spokes in the wheel pattern
    :ivar spoke_radius: Radius of the spokes in the wheel pattern
    :ivar units: Units of the wheel pattern parameters
    """
    center: bool = True
    num_spokes: int = 4
    spoke_radius: float = 1.0 # mm   
    units: str = "mm"

    def get_targets(self, target: geo.Point):
        """
        Get the targets of the focal pattern

        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        if self.center:
            targets = [target.copy()]
            targets[0].id = f"{target.id}_center"
            targets[0].id = f"{target.id} (Center)"
        else:    
            targets = []
        m = target.get_matrix(center_on_point=True)
        for i in range(self.num_spokes):
            theta = 2*np.pi*i/self.num_spokes
            local_position = self.spoke_radius * np.array([np.cos(theta), np.sin(theta), 0.0])
            position = np.dot(m, np.append(local_position, 1.0))[:3]
            spoke = geo.Point(id=f"{target.id}_{np.rad2deg(theta):.0f}deg",
                              name=f"{target.name} ({np.rad2deg(theta):.0f}°)",
                              position=position,
                              units=self.units,
                              radius=target.radius)
            targets.append(spoke)
        return targets
    
    def num_foci(self):
        """
        Get the number of foci in the focal pattern
        
        :returns: Number of foci
        """
        return int(self.center) + self.num_spokes

@dataclass
class SimulationGrid:
    dims: Tuple[str, str, str] = ("lat", "ele", "ax")
    names: Tuple[str, str, str] = ("Lateral", "Elevation", "Axial")
    spacing: float = 1.0
    units: str = "mm"
    x_extent: Tuple[float, float] = (-30., 30.)
    y_extent: Tuple[float, float] = (-30., 30.)
    z_extent: Tuple[float, float] = (-4., 60.)
    dt: float = 0.
    t_end: float = 0.
    c0: float = 1500.0
    cfl: float = 0.5
    options: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        if len(self.dims) != 3:
            raise ValueError("dims must have length 3.")
        if len(self.names) != 3:
            raise ValueError("names must have length 3.")
        if len(self.x_extent) != 2:
            raise ValueError("x_extent must have length 2.")
        if len(self.y_extent) != 2:
            raise ValueError("y_extent must have length 2.")
        if len(self.z_extent) != 2:
            raise ValueError("z_extent must have length 2.")
        self.dims = tuple(self.dims)
        self.names = tuple(self.names)
        nx = np.diff(self.x_extent)/self.spacing
        x_extent = tuple(np.arange(2)*np.round(nx)*self.spacing + self.x_extent[0])
        if ((0.5-np.abs((nx % 1) - 0.5))/ np.round(nx)) > 1e-3:
            logging.warning(f"x_extent {self.x_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {x_extent}.")
        self.x_extent = x_extent
        ny = np.diff(self.y_extent)/self.spacing
        y_extent = tuple(np.arange(2)*np.round(ny)*self.spacing + self.y_extent[0])
        if ((0.5-np.abs((ny % 1) - 0.5))/ np.round(ny)) > 1e-3:
            logging.warning(f"y_extent {self.y_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {y_extent}.")
        self.y_extent = y_extent
        nz = np.diff(self.z_extent)/self.spacing
        z_extent = tuple(np.arange(2)*np.round(nz)*self.spacing + self.z_extent[0])
        if ((0.5-np.abs((nz % 1) - 0.5))/ np.round(nz)) > 1e-3:
            logging.warning(f"z_extent {self.z_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {z_extent}.")
        self.z_extent = z_extent

    def get_coords(self, dims=None, units: Optional[str] = None):
        dims = self.dims if dims is None else dims
        units = self.units if units is None else units
        sizes = self.get_size(dims)
        extents = self.get_extent(dims, units)
        coords = xa.Coordinates({dim: np.linspace(extents[i][0], extents[i][1], sizes[i]) for i, dim in enumerate(dims)})
        for i, dim in enumerate(dims):
            coords[dim].attrs['units'] = units
            coords[dim].attrs['long_name'] = self.names[i]   
        return coords
    
    def get_corners(self, id: str = "corners", units: Optional[str] = None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        xyz = np.array(np.meshgrid(self.x_extent, self.y_extent, self.z_extent, indexing='ij'))
        corners = xyz.reshape(3,-1)
        return corners*scl
    
    def get_extent(self, dims: Optional[str]=None, units: Optional[str] = None):
        dims = self.dims if dims is None else dims
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        extents = [self.x_extent, self.y_extent, self.z_extent]
        return np.array([extents[self.dims.index(dim)] for dim in dims])*scl       
    
    def get_max_cycle_offset(self, arr:xdc.Transducer, frequency: Optional[float] = None, delays: Optional[np.ndarray]=None, zmin: float =10e-3):    
        frequency = arr.frequency if frequency is None else frequency
        delays = np.zeros(arr.numelements()) if delays is None else delays
        coords = self.get_coords(units="m")
        cvals = [coords['lat'], coords['ele'], coords['ax'].sel(ax=slice(zmin, None))]
        ndg = np.meshgrid(*cvals)
        dists = [np.sqrt((ndg[0]-pos[0])**2 + (ndg[1]-pos[1])**2 + (ndg[2]-pos[2])**2) for pos in arr.get_positions(units="m")]
        tof = [dist/self.c0 + delays[i] for i, dist in enumerate(dists)]
        dtof = np.array(tof).max(axis=0) - np.array(tof).min(axis=0)
        max_cycle_offset = dtof.max()*frequency
        return max_cycle_offset

    def get_max_distance(self, arr: Transducer, units: Optional[str] = None):
        units = self.units if units is None else units
        corners = self.get_corners(units=units)
        distances = np.array([[el.distance_to_point(corner, units=units) for corner in corners.T] for el in arr.elements])
        max_distance = np.max(distances)
        return max_distance
    
    def get_size(self, dims: Optional[str]=None):
        dims = self.dims if dims is None else dims
        n = [int(np.round(np.diff(ext)/self.spacing))+1 for ext in [self.x_extent, self.y_extent, self.z_extent]]
        return np.array([n[self.dims.index(dim)] for dim in dims]).squeeze()
    
    def get_spacing(self, units: Optional[str] = None):
        units = self.units if units is None else units
        return getunitconversion(self.units, units)*self.spacing

    def transform_scene(self, scene, id: Optional[str] = None, name: Optional[str] = None, units: Optional[str] = None):
        raise NotImplementedError
    
    @staticmethod
    def from_dict(d):
        return SimulationGrid(**d)

@dataclass
class MaterialReference:
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
        if isinstance(material_id, tuple) or  isinstance(material_id, list):
            return {m: MaterialReference.get_materials(m, as_dict=False) for m in material_id}
        elif material_id == "water":
            m = MaterialReference(id="water",
                                     name="water",
                                     sound_speed=1500.0,
                                     density=1000.0,
                                     attenuation=0.0,
                                     specific_heat=4182.0,
                                     thermal_conductivity=0.598)
        elif material_id == "tissue":
            m = MaterialReference(id="tissue",
                                     name="tissue",
                                     sound_speed=1540.0,
                                     density=1000.0,
                                     attenuation=0.0,
                                     specific_heat=3600.0,
                                     thermal_conductivity=0.5)
        elif material_id == "skull":
            m = MaterialReference(id="skull",
                                     name="skull",
                                     sound_speed=4080.0,
                                     density=1900.0,
                                     attenuation=0.0,
                                     specific_heat=1100.0,
                                     thermal_conductivity=0.3)
        elif material_id == "air":
            m = MaterialReference(id="air",
                                     name="air",
                                     sound_speed=344.0,
                                     density=1.25,
                                     attenuation=0.0,
                                     specific_heat=1012.0,
                                     thermal_conductivity=0.025)
        elif material_id == 'standoff':
            m = MaterialReference(id="standoff",
                                     name="standoff",
                                     sound_speed=1420.0,
                                     density=1000.0,
                                     attenuation=1.0,
                                     specific_heat=4182.0,
                                     thermal_conductivity=0.598)
        else:
            raise ValueError(f"Material {material_id} not found.")
        if as_dict:
            return {m.id: m}
        else:
            return m

    def from_dict(self, d):
        if isinstance(d, list) or isinstance(d, tuple):
            return {dd['id']: MaterialReference.from_dict(dd) for dd in d}
        elif isinstance(str):
            return MaterialReference.get_materials(d, as_dict=False)
        else:
            return MaterialReference(**d)

@dataclass
class DelayMethod(ABC):
    @abstractmethod
    def calc_delays(self, arr: Transducer, target: geo.Point, params: xa.Dataset, transform: bool = True):
        pass

    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d
    
    @staticmethod
    def from_dict(d):
        d = d.copy()
        class_constructor = globals()[d.pop("class")]
        return class_constructor(**d)
    
@dataclass
class DirectDelays(DelayMethod):
    def calc_delays(self, arr: Transducer, target: geo.Point, params: xa.Dataset, transform: bool = True):
        c = params['sound_speed'].attrs['ref_value']
        target_pos = target.get_position(units="m")
        matrix = arr.get_matrix(units="m") if transform else np.eye(4)
        dists = np.array([el.distance_to_point(target_pos, units="m", matrix=matrix) for el in arr.elements])
        tof = dists / c
        #print(tof)
        #tof = np.linalg.norm(target.get_position(units="m") - arr.get_positions(units="m", transform=transform), axis=1) / c
        #print(tof)
        delays = max(tof) - tof
        return delays

@dataclass
class ApodizationMethod:
    @abstractmethod
    def calc_apodization(self, arr: Transducer, target: geo.Point, params: xa.Dataset, transform: bool = True):
        pass

    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d
    
    @staticmethod
    def from_dict(d):
        d = d.copy()
        class_constructor = globals()[d.pop("class")]
        return class_constructor(**d)

@dataclass
class UniformApodization(ApodizationMethod):
    value = 1
    def calc_apodization(self, arr: Transducer, target: geo.Point, params: xa.Dataset, transform: bool = True):
        return np.full(arr.numelements(), self.value)

@dataclass
class MaxAngleApodization(ApodizationMethod):
    max_angle: float = 30.0
    units: str = "deg"
    def calc_apodization(self, arr: Transducer, target: geo.Point, params: xa.Dataset, transform: bool = True):
        target_pos = target.get_position(units="m")
        matrix = arr.get_matrix(units="m") if transform else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        apod[angles <= self.max_angle] = 1
        return apod

@dataclass
class SegmentationMethod:
    materials: dict = field(default_factory=lambda: MaterialReference.get_materials("all"), repr=False)
    ref_material: str = "water"
    
    def __post_init__(self):
        if self.ref_material not in self.materials.keys():
            raise ValueError(f"Reference material {self.ref_material} not found.")
    @abstractmethod
    def _segment(self, volume: xa.DataArray):
        pass

    @staticmethod
    def from_dict(d):
        if isinstance(d, str):
            if d == "water":
                return UniformWater()
            elif d == "tissue":
                return UniformTissue()
            elif d == "segmented":
                return SegmentMRI()
        else:
            d = d.copy()
            class_constructor = globals()[d.pop("class")]
            return class_constructor(**d)

    def _material_indices(self, materials: Optional[dict] = None):
        materials = self.materials if materials is None else materials
        return {material_id: i for i, material_id in enumerate(materials.keys())}
    
    def _map_params(self, seg: xa.DataArray, materials: Optional[dict] = None):
        materials = self.materials if materials is None else materials
        material_dict = self._material_indices(materials=materials)
        params = xa.Dataset()
        ref_mat = materials[self.ref_material]
        for param_id in ref_mat.param_ids:
            info = MaterialReference.param_info(param_id)
            param = xa.DataArray(np.zeros(seg.shape), coords=seg.coords, attrs={"units": info["units"], "long_name": info["name"], "ref_value": ref_mat.get_param(param_id)})
            for material_id, material in materials.items():
                midx = material_dict[material_id]
                param.data[seg.data == midx] = getattr(material, param_id)
            params[param_id] = param
        params.attrs['ref_material'] = ref_mat
        return params
    
    def seg_params(self, volume: xa.DataArray, materials: Optional[dict] = None):
        materials = self.materials if materials is None else materials
        seg = self._segment(volume)
        params = self._map_params(seg, materials=materials)
        return params
    
    def ref_params(self, coords: xa.Coordinates):
       seg = self._ref_segment(coords)
       params = self._map_params(seg)
       return params
    
    def _ref_segment(self, coords: xa.Coordinates):
       material_dict = self._material_indices()
       m_idx = material_dict[self.ref_material]
       sz = list(coords.sizes.values())
       seg = xa.DataArray(np.full(sz, m_idx, dtype=int), coords=coords)
       return seg
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

@dataclass
class UniformSegmentation(SegmentationMethod):
    def _segment(self, vol: xa.DataArray):
       return self._ref_segment(vol.coords)
    
@dataclass
class UniformWater(UniformSegmentation):
    ref_material: str = "water"
    
@dataclass
class UniformTissue(UniformSegmentation):
    ref_material: str = "tissue"

@dataclass
class SegmentMRI(SegmentationMethod):
    def _segment(self, volume: xa.DataArray):
        raise NotImplementedError

@dataclass
class BeamformingPlan:
    id: str = "bf_plan"
    name: str = "Beamforming Plan"
    delay_method: DelayMethod = field(default_factory=DirectDelays)
    apod_method: ApodizationMethod = field(default_factory=UniformApodization)
    seg_method: SegmentationMethod = field(default_factory=UniformWater)

    @staticmethod
    def from_dict(d):
        d = d.copy()
        d["delay_method"] = DelayMethod.from_dict(d.get("delay_method", {}))
        d["apod_method"] = ApodizationMethod.from_dict(d.get("apod_method", {}))
        d["seg_method"] = SegmentationMethod.from_dict(d.get("seg_method", {}))
        return BeamformingPlan(
            id=d['id'],
            name=d['name'],
            delay_method=d["delay_method"],
            apod_method=d["apod_method"],
            seg_method=d["seg_method"])

    def get_ref_params(self, coords: xa.Coordinates):
        return self.seg_method.ref_params(coords)
    
    def beamform(self, arr: xdc.Transducer, target:geo.Point, params: xa.Dataset):
        delays = self.delay_method.calc_delays(arr, target, params)
        apod = self.apod_method.calc_apodization(arr, target, params)
        return delays, apod