from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import plugins.SingleAgentPathPlanningTools as SAPP


def init_plugin():
    singleagentpathplanning = SingleAgentPathPlanning()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SINGLEAGENTPATHPLANNING',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class SingleAgentPathPlanning(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"plugins/SingleAgentPathPlanningTools/model", env=None)
        with traf.settrafarrays():
            traf.target_heading = np.array([])

    @core.timed_function(name='SingleAgentPathPlanning', dt=SAPP.constants.TIMESTEP)
    def update(self):
        for id in traf.id:
            idx = traf.id2idx(id)
            obs = self._get_obs(idx)
            action, _ = self.model.predict(obs, deterministic=True)
            self._set_action(action,idx)
    
    def create(self, n=1):
        super().create(n)
        self.update()
        traf.target_heading[-n:] = traf.hdg[-n:]

    def _get_obs(self, idx):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """
        brg, dis = tools.geo.kwikqdrdist(SAPP.constants.SCHIPHOL[0], SAPP.constants.SCHIPHOL[1], traf.lat[idx], traf.lon[idx])

        x = np.sin(np.radians(brg))*dis*SAPP.constants.NM2KM / SAPP.constants.MAX_DISTANCE
        y = np.cos(np.radians(brg))*dis*SAPP.constants.NM2KM / SAPP.constants.MAX_DISTANCE

        observation = {
            "x" : np.array([x]),
            "y" : np.array([y])
        }
        return observation
    
    def _set_action(self, action, idx):
        bearing = np.rad2deg(np.arctan2(action[0],action[1]))
        traf.target_heading[idx] = bearing
        # traf.ap.selhdgcmd(idx,bearing) # could consider HDG stack command here