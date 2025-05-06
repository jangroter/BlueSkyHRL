from bluesky import traf
from bluesky.tools.aero import kts, ft, fpm
import numpy as np
from openap import FuelFlow


class FuelLogger():    
    def __init__(self):
        self.fuelflow = FuelFlow(ac='A320') # assume all AC are A320
        self.mass = 60000 # assume all ac mass is 60.000

    def get_fuel(self):
        fuel = np.array([self.fuelflow.enroute(mass=self.mass, tas=traf.tas[i]/kts, alt=traf.alt[i]/ft, vs=traf.vs[i]/fpm) for i in traf.id2idx(traf.id)])
        return fuel

