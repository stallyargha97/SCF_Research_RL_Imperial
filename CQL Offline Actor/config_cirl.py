"""CQL CIRL config — actor outputs [Kp,Ki,Kw] (9-D state, action_dim=3).
Kw is the back-calculation anti-windup gain, learned by the actor within a
wide native box (not pinned to the expert constant) so CQL can move the
policy meaningfully away from the expert/BC starting point."""
import numpy as np
from config_common import *

STATE_DIM  = 9
ACTION_DIM = 3
ACTOR_KIND = 'gain'
OBS_LOW  = np.array([ 30.0, -40.0, -40000.0, -5.0,    0.0,  0.0, 10.0,   0.0, 55.0], dtype=np.float32)
OBS_HIGH = np.array([100.0,  55.0,  40000.0,  5.0, 1200.0, 50.0, 90.0, 180.0, 85.0], dtype=np.float32)
GAIN_LOW  = np.array([-3.5, -0.060, -0.35], dtype=np.float32)
GAIN_HIGH = np.array([-0.1, -0.0002, 0.05], dtype=np.float32)
START_GAIN = np.array([KP_EXPERT, KI_EXPERT, KI_EXPERT / KP_EXPERT], dtype=np.float32)
