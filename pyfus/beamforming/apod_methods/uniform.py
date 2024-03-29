from dataclasses import dataclass
import xarray as xa
import numpy as np
from pyfus import Transducer, Point
from pyfus.beamforming.apod_methods import ApodizationMethod

@dataclass
class Uniform(ApodizationMethod):
    value = 1
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform: bool = True):
        return np.full(arr.numelements(), self.value)