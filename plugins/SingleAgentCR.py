from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import plugins.SingleAgentCRTools as SACR

# TODO make this such that you can select the algorithm in settings
settings.set_variable_defaults(SingleAgentCR_alg='SB3-SAC')

def init_plugin():
    singleagentconflictresolution = SingleAgentConflictResolution()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SINGLEAGENTCONFLICTRESOLUTION',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class SingleAgentConflictResolution(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"plugins/SingleAgentCRTools/model", env=None)
        with traf.settrafarrays():
            traf.target_heading = np.array([])

    @core.timed_function(name='SingleAgentConflictResolution', dt=SACR.constants.TIMESTEP)
    def update(self):
        for id in traf.id:
            idx = traf.id2idx(id)
            obs = self._get_obs(idx)
            clipped_obs = {key: np.clip(arr, -1.2, 1.2) for key, arr in obs.items()}
            action, _ = self.model.predict(clipped_obs, deterministic=True)
            self._set_action(action,idx)
    
    def create(self, n=1):
        super().create(n)
        traf.target_heading[-n:] = traf.hdg[-n:]

    def _get_obs(self, idx):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """
        # x_r = np.ones(SACR.constants.NUM_AC_STATE)*10
        # y_r = np.ones(SACR.constants.NUM_AC_STATE)*10
        # vx_r = np.zeros(SACR.constants.NUM_AC_STATE)
        # vy_r = np.zeros(SACR.constants.NUM_AC_STATE)
        # cos_track = np.zeros(SACR.constants.NUM_AC_STATE)
        # sin_track = np.zeros(SACR.constants.NUM_AC_STATE)
        # distances = np.ones(SACR.constants.NUM_AC_STATE)*10
        x_r = np.array([])
        y_r = np.array([])
        vx_r = np.array([])
        vy_r = np.array([])
        cos_track = np.array([])
        sin_track = np.array([])
        distances = np.array([])
        # Drift of agent aircraft for reward calculation
        drift = 0

        ac_hdg = traf.hdg[idx]
        target_hdg = traf.target_heading[idx]

        drift = ac_hdg - target_hdg
        drift = SACR.functions.bound_angle_positive_negative_180(drift)
        
        cos_drift = np.array(np.cos(np.deg2rad(drift)))
        sin_drift = np.array(np.sin(np.deg2rad(drift)))

        # Get agent aircraft airspeed, m/s
        airspeed = np.array(traf.tas[idx])

        vx = np.cos(np.deg2rad(ac_hdg)) * traf.gs[idx]
        vy = np.sin(np.deg2rad(ac_hdg)) * traf.gs[idx]

        ac_loc = SACR.functions.latlong_to_nm(SACR.constants.CENTER, np.array([traf.lat[idx], traf.lon[idx]])) * SACR.constants.NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
        dist = [SACR.functions.euclidean_distance(ac_loc, SACR.functions.latlong_to_nm(SACR.constants.CENTER, np.array([traf.lat[i], traf.lon[i]])) * SACR.constants.NM2KM * 1000) for i in range(len(traf.id))]
        ac_idx_by_dist = np.argsort(dist)

        for i in range(len(traf.id)):
            int_idx = ac_idx_by_dist[i]
            if int_idx == idx:
                continue
            int_hdg = traf.hdg[int_idx]
            
            # Intruder AC relative position, m
            int_loc = SACR.functions.latlong_to_nm(SACR.constants.CENTER, np.array([traf.lat[int_idx], traf.lon[int_idx]])) * SACR.constants.NM2KM * 1000
            x_r = np.append(x_r, int_loc[0] - ac_loc[0])
            y_r = np.append(y_r, int_loc[1] - ac_loc[1])
            # Intruder AC relative velocity, m/s
            vx_int = np.cos(np.deg2rad(int_hdg)) * traf.gs[int_idx]
            vy_int = np.sin(np.deg2rad(int_hdg)) * traf.gs[int_idx]
            vx_r = np.append(vx_r, vx_int - vx)
            vy_r = np.append(vy_r, vy_int - vy)

            # Intruder AC relative track, rad
            track = np.arctan2(vy_int - vy, vx_int - vx)
            cos_track = np.append(cos_track, np.cos(track))
            sin_track = np.append(sin_track, np.sin(track))

            distances = np.append(distances, dist[int_idx])


        observation = {
            "cos(drift)": np.array([cos_drift]),
            "sin(drift)": np.array([sin_drift]),
            "airspeed": np.array([(airspeed-150)/6]),
            "x_r": x_r[:SACR.constants.NUM_AC_STATE]/13000,
            "y_r": y_r[:SACR.constants.NUM_AC_STATE]/13000,
            "vx_r": vx_r[:SACR.constants.NUM_AC_STATE]/32,
            "vy_r": vy_r[:SACR.constants.NUM_AC_STATE]/66,
            "cos(track)": cos_track[:SACR.constants.NUM_AC_STATE],
            "sin(track)": sin_track[:SACR.constants.NUM_AC_STATE],
            "distances": (distances[:SACR.constants.NUM_AC_STATE]-50000.)/15000.
        }

        return observation
    
    def _set_action(self, action, idx):
        dh = action[0] * SACR.constants.D_HEADING
        dv = action[1] * SACR.constants.D_VELOCITY
        heading_new = SACR.functions.bound_angle_positive_negative_180(traf.hdg[idx] + dh)
        speed_new = (traf.cas[idx] + dv) * SACR.constants.MpS2Kt

        id = traf.id[idx]
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")