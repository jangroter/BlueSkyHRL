""" 
quick comment on the units: 

time - Unix timestamp: The time at which the state vector was valid.
icao24 - string: The unique 24-bit transponder ID assigned to the aircraft.
lat, lon - double: The last known latitude and longitude (WGS84 format).
velocity - double: Speed over ground in meters per second.
heading - double: The direction of movement in degrees from geographic north.
vertrate - double: Vertical speed in meters per second (positive = ascending, negative = descending).
callsign - string: The flight identifier broadcast by the aircraft.
onground - boolean: Indicates if the aircraft is on the ground (true) or airborne (false).
alert, spi - boolean: Special ATC indicators (alert squawk, special position indicator).
squawk - string: The 4-digit octal transponder code assigned by ATC.
baroaltitude - double: Barometric altitude measured by the aircraft.
geoaltitude - double: Geometric altitude determined by GNSS (GPS).
lastposupdate - double: The timestamp of the last recorded position update.
lastcontact - double: The last time OpenSky received a signal from the aircraft.
hour - int: The start of the hour this data belongs to.

The logger should log every 1 second, this creates a huge dataset, but that is fine for processing.
Additionally maybe 

"""

from bluesky import core, stack, traf, tools, settings, sim
from stable_baselines3 import SAC
import numpy as np
import pandas as pd
import plugins.MultiAgentCRTools as MACR
import torch

SAVE_INTERVAL = 600 # every 10 minutes
FOLDER = 'output/experiment'


def init_plugin():
    flightlogger = FlightLogger()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'FLIGHTLOGGER',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class FlightLogger(core.Entity):
    def __init__(self):
        super().__init__()
        # create numpy arrays for each column
        self.time = []
        self.icao24 = []
        self.lat = np.array([])
        self.lon = np.array([])
        self.velocity = np.array([])
        self.heading = np.array([])
        self.vertrate = np.array([])
        self.callsign = []
        self.onground = []
        self.alert = []
        self.spi = []
        self.squawk = []
        self.baroaltitude = np.array([])
        self.geoaltitude = np.array([])
        self.lastposupdate = []
        self.lastcontact = []
        self.serials = np.array([]) # unsure what to log here
        self.hour = []

    @core.timed_function(name='FlightLogger', dt=1)
    def update(self):
        self.time.extend([sim.utc] * len(traf.id))
        self.icao24.extend(traf.id)

        self.lat = np.append(self.lat, traf.lat)
        self.lon = np.append(self.lon, traf.lon)
        self.velocity = np.append(self.velocity, traf.gs)
        self.heading = np.append(self.heading, traf.hdg)
        self.vertrate = np.append(self.vertrate, traf.vs)

        self.callsign.extend(traf.id)
        self.onground.extend([False] * len(traf.id))
        self.alert.extend([False] * len(traf.id))
        self.spi.extend([False] * len(traf.id))
        self.squawk.extend([False] * len(traf.id))

        self.baroaltitude = np.append(self.baroaltitude, traf.alt)
        self.geoaltitude = np.append(self.geoaltitude, traf.alt)

        self.lastposupdate.extend([sim.utc] * len(traf.id))
        self.lastcontact.extend([sim.utc] * len(traf.id))
        
        self.serials = np.append(self.serials, traf.lat)
        self.hour.extend([sim.utc.replace(minute=0, second=0, microsecond=0)] * len(traf.id))

    @core.timed_function(dt=SAVE_INTERVAL)
    def save(self):
        save_data = pd.DataFrame({
            "time": self.time,
            "icao24": self.icao24,
            "lat": self.lat,
            "lon": self.lon,
            "velocity": self.velocity,
            "heading": self.heading,
            "vertrate": self.vertrate,
            "callsign": self.callsign,
            "onground": self.onground,
            "alert": self.alert,
            "spi": self.spi,
            "squawk": self.squawk,
            "baroaltitude": self.baroaltitude,
            "geoaltitude": self.geoaltitude,
            "lastposupdate": self.lastposupdate,
            "lastcontact": self.lastcontact,
            "hour": self.hour
        })

        save_data.to_csv(f'{FOLDER}/flight_output.csv', sep=",")