from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import plugins.MultiAgentCRattTools as MACRT
import torch

def init_plugin():
    multiagentconflictresolutionattention = MultiAgentConflictResolutionAttention()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MULTIAGENTCONFLICTRESOLUTIONATTENTION',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class MultiAgentConflictResolutionAttention(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = MACRT.actor.MultiHeadAdditiveActorBasic(q_dim = 3,
                                                            kv_dim = 7,
                                                            out_dim = 2,
                                                            num_heads = 5,
                                                            dropout_rate=0)
        self.model.load_state_dict(torch.load("plugins/MultiAgentCRattTools/weights/actor.pt"))
        self.model.set_test(True)
        with traf.settrafarrays():
            traf.target_heading = np.array([])

    @core.timed_function(name='MultiAgentConflictResolutionAttention', dt=MACRT.constants.TIMESTEP)
    def update(self):
        if len(traf.id) > 0:
            observations = self._get_obs()
            obs_array = np.array(list(observations.values()))
            # obs_array = np.clip(obs_array,-5,5)
            action = self.model(torch.FloatTensor(np.array([obs_array])))
            action = np.array(action[0].detach().numpy())
            act_array = np.clip(action, -1, 1)

            for id, action in zip(traf.id,act_array[0]):
                idx = traf.id2idx(id)
                self._set_action(action,idx)
        else:
            pass
    
    def create(self, n=1):
        super().create(n)
        traf.target_heading[-n:] = traf.hdg[-n:]

    def _get_obs(self):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """
        obs = []

        for id in traf.id:
            ac_idx = traf.id2idx(id)
            ac_hdg = traf.hdg[ac_idx]
            
            # Get and decompose agent aircaft drift
            ac_hdg = traf.hdg[ac_idx]
            target_hdg = traf.target_heading[ac_idx]

            drift = ac_hdg - target_hdg
            drift = MACRT.functions.bound_angle_positive_negative_180(drift)
            cos_drift = np.cos(np.deg2rad(drift))
            sin_drift = np.sin(np.deg2rad(drift))
        
            # Get agent aircraft airspeed, m/s
            airspeed = traf.tas[ac_idx]

            vx = np.cos(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]

            ac_loc = MACRT.functions.latlong_to_nm(MACRT.constants.CENTER, np.array([traf.lat[ac_idx], traf.lon[ac_idx]])) * MACRT.constants.NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
            x = ac_loc[0]
            y = ac_loc[1]

            observation = {
                "cos(drift)": np.array([cos_drift]),
                "sin(drift)": np.array([sin_drift]),
                "airspeed": np.array([(airspeed-150)/50]),
                "x": np.array([x/50000]),
                "y": np.array([y/50000]),
                "vx": np.array([vx/150]),
                "vy": np.array([vy/150])
            }

            obs.append(np.concatenate(list(observation.values())))

        observations = {
            a: o
            for a, o in zip(traf.id, obs)
        }

        return observations
    
    def _set_action(self, action, idx):
        dh = action[0] * MACRT.constants.D_HEADING
        dv = action[1] * MACRT.constants.D_VELOCITY
        heading_new = MACRT.functions.bound_angle_positive_negative_180(traf.hdg[idx] + dh)
        speed_new = (traf.cas[idx] + dv) * MACRT.constants.MpS2Kt

        id = traf.id[idx]
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")