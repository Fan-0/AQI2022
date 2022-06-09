import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qiskit as qk
from IPython.display import display
from qiskit import IBMQ, pulse
from qiskit.circuit import Parameter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.visualization import plot_bloch_multivector

warnings.filterwarnings('ignore')
from qiskit.tools.jupyter import *

token = 'b6464d13b284902ed1d1a48d2aed6bd0474c7be45011741b0fb879614419659cb722e74a046af3d5caae0398aec9bdac8843068ecbee91aff466cf3e30f3bef5'
try:
    IBMQ.load_account()
except:
    qk.IBMQ.save_account(token=token)
    qk.IBMQ.enable_account(token)
provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
backend = provider.get_backend("ibmq_armonk")## 

# samples need to be multiples of 16
def get_dt_from(sec):
    return get_closest_multiple_of_16(sec/dt)

# Convert seconds to dt
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

# center data around 0
def baseline_remove(values):
    return np.array(values) - np.mean(values)

def Probe_Pulse(test_Pulse,drive_amp,num_rabi_points):
    drive_amp_min = -1
    drive_amp_max = 1
    drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

    with pulse.build(backend=backend, default_alignment='sequential', name='Rabi Experiment') as rabi_sched:
        #pulse.set_frequency(rough_qubit_frequency, 0)
        pulse.call(test_Pulse)
        
        pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
    
    rabi_schedules = [rabi_sched.assign_parameters({drive_amp: a}, inplace=False) for a in drive_amps]
    return rabi_schedules[0]


