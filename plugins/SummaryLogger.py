"""
Logger plugin that for each flight logs the number of unique LoS, total noise exposure and total emisions
"""

from bluesky import core, stack, traf, tools, settings, sim
from stable_baselines3 import SAC
import numpy as np
import pandas as pd
import torch

from plugins.LoggerTools import noise_logger
from plugins.LoggerTools import fuel_logger

SAVE_INTERVAL = 600 # every 10 minutes
FOLDER = 'output/experiment'

def init_plugin():
    summarylogger = SummaryLogger()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SUMMARYLOGGER',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class SummaryLogger(core.Entity):
    def __init__(self):
        super().__init__()
        self.noise_logger = noise_logger.NoiseLogger()
        self.fuel_logger = fuel_logger.FuelLogger()

        # for intrusions, look at traf.cd.lospairs_all & traf.cd.lospairs_unique

        columns = ["ACID", "total_noise", "total_fuel", "flight_time"]
        self.data = pd.DataFrame(columns=columns)
        self.total_intrusions = 0
        self.total_conflicts = 0

    @core.timed_function(dt=1)
    def update(self):

        noise = self.get_noise()
        fuel = self.get_fuel()

        new_data = pd.DataFrame({
            'ACID': traf.id,
            'total_noise': noise,
            'total_fuel': fuel,
            'flight_time': [1]*len(traf.id)
        })

        merged_df = self.data.merge(new_data, on='ACID', how='outer', suffixes=('_old', '_new'))

        merged_df['total_noise_old'].fillna(0, inplace=True)
        merged_df['total_fuel_old'].fillna(0, inplace=True)
        merged_df['flight_time_old'].fillna(0, inplace=True)

        merged_df['total_noise_new'].fillna(0, inplace=True)
        merged_df['total_fuel_new'].fillna(0, inplace=True)
        merged_df['flight_time_new'].fillna(0, inplace=True)

        merged_df['total_noise'] = merged_df['total_noise_old'] + merged_df['total_noise_new']
        merged_df['total_fuel'] = merged_df['total_fuel_old'] + merged_df['total_fuel_new']
        merged_df['flight_time'] = merged_df['flight_time_old'] + merged_df['flight_time_new']

        self.data = merged_df[['ACID', 'total_noise', 'total_fuel', 'flight_time']].sort_values('ACID').reset_index(drop=True)

        self.total_intrusions = len(traf.cd.lospairs_all)
        self.total_conflicts = len(traf.cd.confpairs_all)

    @core.timed_function(dt=SAVE_INTERVAL)
    def save(self):
        self.data.to_csv(f'{FOLDER}/summary.csv', sep=',')
        np.savetxt(f'{FOLDER}/intrusions.csv',np.array([self.total_intrusions]))
        np.savetxt(f'{FOLDER}/conflicts.csv',np.array([self.total_conflicts]))

    def get_noise(self):
        noise = self.noise_logger.get_noise()
        return noise

    def get_fuel(self):
        fuel = self.fuel_logger.get_fuel()
        return fuel

    def get_intrusions(self):
        pass
