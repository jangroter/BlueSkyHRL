import numpy as np

# Conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FL2M = 30.48

INTRUSION_DISTANCE = 5 # NM

# Model parameters
NUM_AC_STATE = 4
D_HEADING = 22.5 # deg
D_VELOCITY = 20/3 # kts

TIMESTEP = 5

CENTER = np.array([51.990426702297746, 4.376124857109851]) # TU Delft AE Faculty coordinates