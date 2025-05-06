from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import kts, ft

import plugins.CommonTools.common as common
import plugins.CommonTools.functions as fn
import plugins.FlightEnvelope as fe

import numpy as np

settings.set_variable_defaults(TrafficDemandLevel=60) # aircraft per hour
settings.set_variable_defaults(TotalFlights=10.000) # total number of flights to be generated in the scenario file


DT = 1 # check for new aircraft every second

def init_plugin():
    trafficgenerator = TrafficGenerator()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'TRAFFICGENERATOR',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class TrafficGenerator(core.Entity):  
    """
    Traffic generator for the scenarios in chapters 6, 7 and 8 on the HRL based Arrival Manager
    
    Uses an exponential distribution to obtain the time between aircraft entering the airspace
    to mimic the Poisson Process nature of the problem.
    
    Aircraft are initialized uniformly along the border of the environment
    """
    def __init__(self):
        super().__init__()
        self.tdl = settings.TrafficDemandLevel  # aircraft per hour
        self.mean_seconds_per_ac = 3600 / self.tdl

        self.total_flights = settings.TotalFlights
        self.flights_spawned = 0

        self._get_next_arrival()
        self.time_elapsed = 0


    @core.timed_function(name='TrafficGenerator', dt=DT)
    def update(self):
        if self.time_elapsed > self.next_arrival:
            self._spawn_aircraft()
            self._get_next_arrival()
        self.time_elapsed += DT
        
        vmin, vmax, _, _ = traf.perf.currentlimits()
        print(vmin)
        print(vmax)
        print(traf.alt)
        print(traf.cas)

    def _get_next_arrival(self):
        self.next_arrival = np.random.exponential(scale=self.mean_seconds_per_ac)
        self.time_elapsed = 0
        
    def _spawn_aircraft(self):
        lat, lon, heading, altitude = self._get_spawn()
        altitude_ft = altitude / ft
        speed = fe.get_speed_at_altitude(altitude) / kts
        stack.stack(f"CRE KL{self.flights_spawned} A320 {lat} {lon} {heading} {altitude_ft} {speed}")
        self.flights_spawned += 1


    def _get_spawn(self):
        """
        
        """
        spawn_bearing = np.random.uniform(0,360)
        spawn_distance = common.airspace_radius

        spawn_lat, spawn_lon = fn.get_point_at_distance(common.schiphol[0],common.schiphol[1],spawn_distance,spawn_bearing)
        spawn_heading = (spawn_bearing + 180 + 360)%360

        spawn_altitude = np.random.uniform(common.altitude_min,common.altitude_max)

        return spawn_lat, spawn_lon, spawn_heading, spawn_altitude


    