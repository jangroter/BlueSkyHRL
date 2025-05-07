from bluesky import core, stack, traf, tools, settings 
import plugins.CommonTools.functions as fn
from matplotlib.path import Path
import numpy as np


RUNWAYS_SCHIPHOL_FAF = {
        "18C": {"lat": 52.301851, "lon": 4.737557, "track": 183},
        "36C": {"lat": 52.330937, "lon": 4.740026, "track": 3},
        "18L": {"lat": 52.291274, "lon": 4.777391, "track": 183},
        "36R": {"lat": 52.321199, "lon": 4.780119, "track": 3},
        "18R": {"lat": 52.329170, "lon": 4.708888, "track": 183},
        "36L": {"lat": 52.362334, "lon": 4.711910, "track": 3},
        "06":   {"lat": 52.304278, "lon": 4.776817, "track": 60},
        "24":   {"lat": 52.288020, "lon": 4.734463, "track": 240},
        "09":   {"lat": 52.318362, "lon": 4.796749, "track": 87},
        "27":   {"lat": 52.315940, "lon": 4.712981, "track": 267},
        "04":   {"lat": 52.313783, "lon": 4.802666, "track": 45},
        "22":   {"lat": 52.300518, "lon": 4.783853, "track": 225}
    }


RUNWAY = ['27','18R']

FAF_DISTANCE = 10 #km
IAF_DISTANCE = 25 #km, from FAF

IAF_ANGLE = 90 #degrees, symmetrical around FAF

def init_plugin():
    sink = Sink()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SINK',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class Sink(core.Entity):  
    def __init__(self):
        super().__init__()
        self.runway = RUNWAY
        self._set_terminal_conditions()
        with self.settrafarrays():
             self.last_lat = np.array([])
             self.last_lon = np.array([])

    @core.timed_function(name='Sink', dt=5)
    def update(self):
        for id in traf.id:
            self._get_terminated(id)
        self._update_positions()
    
    def create(self, n=1):
        super().create(n)
        self.last_lat[-n:] = traf.lat[-n:]
        self.last_lon[-n:] = traf.lon[-n:]

    def _set_terminal_conditions(self):
            """
            Creates the terminal conditions surrounding the FAF to ensure correct approach angle

            If render mode is set to 'human' also already creates the required elements for plotting 
            these terminal conditions in the rendering window
            """
            for rwy in self.runway:
                num_points = 36 # number of straight line segments that make up the circle

                faf_lat, faf_lon = fn.get_point_at_distance(RUNWAYS_SCHIPHOL_FAF[rwy]['lat'],
                                                            RUNWAYS_SCHIPHOL_FAF[rwy]['lon'],
                                                            FAF_DISTANCE,
                                                            RUNWAYS_SCHIPHOL_FAF[rwy]['track']-180)
                
                # Compute bounds for the merge angles from FAF
                cw_bound = ((RUNWAYS_SCHIPHOL_FAF[rwy]['track']-180+ 360)%360) + (IAF_ANGLE/2)
                ccw_bound = ((RUNWAYS_SCHIPHOL_FAF[rwy]['track']-180+ 360)%360) - (IAF_ANGLE/2)

                angles = np.linspace(cw_bound,ccw_bound,num_points)
                lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

                command = f'POLYLINE SINK{rwy}'
                for i in range(0,len(lat_iaf)):
                    command += ' '+str(lat_iaf[i])+' '
                    command += str(lon_iaf[i])
                stack.stack(command)
            
                stack.stack(f'POLYLINE RESTRICT{rwy} {lat_iaf[0]} {lon_iaf[0]} {faf_lat} {faf_lon} {lat_iaf[-1]} {lon_iaf[-1]}')
                stack.stack(f'COLOR RESTRICT{rwy} red')

    def _get_terminated(self, id):
        """
        Checks if the aircraft has passed the IAF beacon and can be routed to the FAF (SINK)
        or if it has missed approach by coming in with a too high turn radius requirements (RESTRICT)
        """
        idx = traf.id2idx(id)
        shapes = tools.areafilter.basic_shapes
        line_ac = Path(np.array([[self.last_lat[idx], self.last_lon[idx]],[traf.lat[idx], traf.lon[idx]]]))
        for rwy in self.runway:
            line_sink = Path(np.reshape(shapes[f'SINK{rwy}'].coordinates, (len(shapes[f'SINK{rwy}'].coordinates) // 2, 2)))
            # line_restrict = Path(np.reshape(shapes['RESTRICT'].coordinates, (len(shapes['RESTRICT'].coordinates) // 2, 2)))

            if line_sink.intersects_path(line_ac):
                stack.stack(f'DEL {id}')

    def _update_positions(self):
        self.last_lat = traf.lat
        self.last_lon = traf.lon

