import numpy as np
from bluesky import traf, tools

SCHIPHOL = [52.3068953,4.760783] # lat,lon coords of schiphol for reference to x_array and y_array
NM2KM = 1.852
NM2M = 1852.

class NoiseLogger():    
    def __init__(self):
        self.pop_array = np.genfromtxt('plugins/LoggerTools/population_1km.csv', delimiter = ' ')
        self.x_array = np.genfromtxt('plugins/LoggerTools/x_array.csv', delimiter = ' ')
        self.y_array = np.genfromtxt('plugins/LoggerTools/y_array.csv', delimiter = ' ')
        self.x_max = np.max(self.x_array)
        self.y_max = np.max(self.y_array)
        self.cell_size = 1000 # distance per pixel in pop_array, in m
        self.projection_size = 30 # distance in km that noise is projected down, similar to kernel size in CNN

    def get_noise(self):
        noise = np.array([self._get_population_exposure(traf.lat[i],traf.lon[i],traf.alt[i]) for i in traf.id2idx(traf.id)])
        return noise

    def _get_population_exposure(self, lat, lon, alt):
        """
        Calculates the population exposed to the aircraft, scaled with the inverse square 
        of the distance. Inverse square of the distance is based on the inverse square law
        of noise dissipation.

        This function assumes that 2 people exposed to 500 units of noise would be equivalent to 
        1 person being exposed to 1000 units of noise. Can be replaced by a more accurate noise
        cost function, but is deemed enough showcasing this environment. 
        """
        brg, dist = tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], lat, lon)

        x = np.sin(np.radians(brg))*dist*NM2M
        y = np.cos(np.radians(brg))*dist*NM2M
        z = alt

        x_index_min = int(((x+self.x_max)/self.cell_size)-self.projection_size)
        x_index_max = int(((x+self.x_max)/self.cell_size)+self.projection_size)
        y_index_min = int(((self.y_max - y)/self.cell_size)-self.projection_size)
        y_index_max = int(((self.y_max - y)/self.cell_size)+self.projection_size)

        distance2 = (self.x_array[y_index_min:y_index_max,x_index_min:x_index_max]-x)**2 + (self.y_array[y_index_min:y_index_max,x_index_min:x_index_max]-y)**2 + z**2
        return np.sum(self.pop_array[y_index_min:y_index_max,x_index_min:x_index_max]/distance2)


