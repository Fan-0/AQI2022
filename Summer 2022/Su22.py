import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import qiskit as qk
from IPython.display import display
from qiskit import IBMQ, pulse
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.visualization import plot_bloch_multivector
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit.providers.aer import PulseSimulator
from qiskit.compiler import assemble
from scipy.optimize import curve_fit
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

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)
    return fitparams, y_fit
    
class Custom_Fgp:
    def __init__(self, name, inp,backend):
        self.name=name
        self.input = np.array(inp)
        self.backend = backend
        drive_sigma_sec = 0.015 * 1.0e-6                          # This determines the actual width of the gaussian
        drive_duration_sec = drive_sigma_sec   
        #self.norm = self.input/np.sqrt((self.input**2).sum())
        self.norm = self.input/self.input.max()
        self.par = Parameter('drive_amp')
        self.length = 1
        self.pi_p,plt = self.full_cal()
    
    #replaced using Customize_pulse_2)
    def Create_Pulse(self):
        temp = []
        for i in self.norm:
            for j in range(int(self.length)):
                temp.append(i)
        temp = np.array(temp)
        with pulse.build(backend=backend, default_alignment='sequential', name='Rabi Experiment') as custom_Pulse:
            [pulse.play(temp*self.pi_p, pulse.drive_channel(0))]
        return custom_Pulse
    
    def Customize_pulse(self,x):
        with pulse.build(backend=backend, default_alignment='sequential', name='Rabi Experiment') as custom_Pulse:
            [pulse.play(self.norm*x, pulse.drive_channel(0))]
            pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        return custom_Pulse
    
    def Customize_pulse_2(self, x, length):
        temp = []
        for i in self.norm:
            for j in range(int(length)):
                temp.append(i)
        temp = np.array(temp)
        with pulse.build(backend=backend, default_alignment='sequential', name='Rabi Experiment') as custom_Pulse:
            [pulse.play(temp*x, pulse.drive_channel(0))]
            pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        return custom_Pulse
    
    def draw(self):
        return self.Create_Pulse_2(1,self.length).draw(backend=self.backend)
    
    def baseline_remove(self, values):
        return np.array(values) - np.mean(values)
    
    '''def rabi_test(self,num_rabi_points): #non functional
        scale_factor = 1e-15
        drive_amp_min = -1
        drive_amp_max = 1
        drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

        rabi_schedules = [self.cont.assign_parameters({self.par: a}, inplace=False) for a in drive_amps]
        num_shots_per_point = 1024
        job = backend.run(rabi_schedules, 
                  meas_level=1, 
                  meas_return='avg', 
                  shots=num_shots_per_point)
        job_monitor(job)
        
        rabi_results = job.result(timeout=120)
        rabi_values = []
        for i in range(num_rabi_points):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(rabi_results.get_memory(i)[0] * scale_factor)

        rabi_values = np.real(self.baseline_remove(rabi_values))

        return drive_amps,rabi_values'''
    
    def Cali(self,num_rabi_points):
        scale_factor = 1e-15
        drive_amp_min = -1
        drive_amp_max = 1
        drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

        rabi_schedules = [self.Customize_pulse_2(a,self.length) for a in drive_amps]
        #return rabi_schedules
        num_shots_per_point = 1024
        job = backend.run(rabi_schedules, 
                  meas_level=1, 
                  meas_return='avg', 
                  shots=num_shots_per_point)
        job_monitor(job)
        
        rabi_results = job.result(timeout=120)
        rabi_values = []
        for i in range(num_rabi_points):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(rabi_results.get_memory(i)[0] * scale_factor)

        rabi_values = np.real(self.baseline_remove(rabi_values))

        return drive_amps,rabi_values
    
    def full_cal(self):
        scale_factor = 1e-15
        num_rabi_points = 10
        drive_amp_min = -1
        drive_amp_max = 1
        drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)
        pi_amp = 2
        counter = 0
        while pi_amp > 0.9:
            counter+=1
            if counter==100:
                break
            drive_amps,rabi_values = self.rabi_test_Sim(num_rabi_points,counter)
            fit_params, y_fit = fit_function(drive_amps,
                                 rabi_values, 
                                 lambda x, A, B, drive_period, phi:(A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                 [-8, 1, 2, 0])
            drive_period = fit_params[2]
            pi_amp = abs(drive_period / 2)
            print(counter, ": ", pi_amp)
        self.length = counter
        drive_amps,rabi_values = self.Cali(50)
        fit_params, y_fit = fit_function(drive_amps,
                                 rabi_values, 
                                 lambda x, A, B, drive_period, phi:(A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                 [-8, 1, 2, 0])
        drive_period = fit_params[2]
        pi_amp = abs(drive_period / 2)
        plt.scatter(drive_amps, rabi_values, color='black')
        plt.plot(drive_amps, y_fit, color='red')
        print(fit_params)
        drive_period = fit_params[2] # get period of rabi oscillation

        plt.axvline(0, color='red', linestyle='--')
        plt.axvline(drive_period/2, color='red', linestyle='--')
        plt.annotate("", xy=(0, 0), xytext=(drive_period/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
        plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')

        plt.xlabel("Drive len [dt]", fontsize=15)
        plt.title("Armonk", fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.show()
        return pi_amp,plt
    
    def Cali_l(self,len_max):
        scale_factor = 1e-15
        drive_len = np.linspace(1,len_max,len_max-1)

        rabi_schedules = [self.Customize_pulse_2(1,a) for a in drive_len]
        #return rabi_schedules
        num_shots_per_point = 1024
        job = backend.run(rabi_schedules, 
                  meas_level=1, 
                  meas_return='avg', 
                  shots=num_shots_per_point)
        job_monitor(job)
        
        rabi_results = job.result(timeout=120)
        rabi_values = []
        for i in range(len_max-1):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(rabi_results.get_memory(i)[0] * scale_factor)

        rabi_values = np.real(self.baseline_remove(rabi_values))

        return drive_len,rabi_values
        
    def rabi_test_Sim(self, num_rabi_points,length):
        scale_factor = 1e-15
        drive_amp_min = -1
        drive_amp_max = 1
        drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

        rabi_schedules = [self.Customize_pulse_2(a,length) for a in drive_amps]
        #return rabi_schedules
        num_shots_per_point = 1024
        armonk_model = PulseSystemModel.from_backend(self.backend)
        backend_sim = PulseSimulator(system_model=armonk_model)

        rabi_qobj = assemble(rabi_schedules, 
                             backend=backend_sim,
                             meas_level=1, 
                             meas_return='avg',
                             shots=num_shots_per_point)
        sim_result = backend_sim.run(rabi_qobj).result()
        rabi_values = []
        for i in range(num_rabi_points):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(sim_result.get_memory(i)[0] * scale_factor)

        rabi_values = np.real(self.baseline_remove(rabi_values))
        return drive_amps,rabi_values
    
    def add_as_gate(self,circ,qubits):
        custom_gate = Gate(self.name, 1,[])
        [circ.append(custom_gate, [i]) for i in qubits]
        circ.add_calibration(self.name, qubits, self.Create_Pulse(), [])
        return circ
    
    def State_vec_tomography():
        return 0
    def Full_tomography():
        return 0