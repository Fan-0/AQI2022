import warnings

import numpy as np
import matplotlib
import random
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
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
from matplotlib import animation
from qiskit.visualization import visualize_transition
from mpl_toolkits.mplot3d import Axes3D
from qiskit.visualization.bloch import Bloch
#from IPython.display import HTML
from qiskit.visualization import plot_bloch_vector
from qiskit import transpile, schedule as build_schedule
import qiskit.result.result as resultifier
import scipy.signal as si
import scipy.linalg as la
import matplotlib.pyplot as plt
from qiskit import Aer
import datetime
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.visualization.pulse_v2 import device_info, stylesheet
from qiskit.visualization.pulse_v2.events import ChannelEvents
from qiskit.visualization.pulse_v2.generators import gen_filled_waveform_stepwise
import  qiskit.pulse.transforms.canonicalization as canon
from joblib import Parallel, delayed
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import ConfigurableFakeBackend,FakeArmonkV2,FakeOpenPulse2Q
from qiskit.pulse.transforms import block_to_schedule
import os
import pickle

# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import qiskit.ignis.mitigation.measurement as mc
import time
test_set= [1,0.5,0.25,2]
    
#defenitions for spectroscopy
cpu_count = 2
noise_power = 1e-3
num_noise_trajs = 30
shots = 100 
backend = FakeOpenPulse2Q()#ConfigurableFakeBackend("memer",1)

#Pickle fuctions
def loadData(inp):
    # for reading also binary mode is important
    dbfile = open(inp, 'rb')     
    db = pickle.load(dbfile)
    dbfile.close()
    return db

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)
    error = np.sqrt(np.diag(conv))
    return fitparams, y_fit, error

class spec_data:
    def __init__(self,all_probs,circ_batch,name=datetime.datetime.now()):
        self.all_probs = all_probs
        self.circ_batch =circ_batch
        self.name = name
    
    def dump(self):
        data = {"spec_data",self.all_probs,self.circ_batch}
        file = open(self.name, 'wb')
        pickle.dump(data,file)
        file.close()
        
        
    
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
        with pulse.build(backend=self.backend, default_alignment='sequential', name='Rabi Experiment') as custom_Pulse:
            [pulse.play(temp*self.pi_p, pulse.drive_channel(0))]
        return custom_Pulse
    
    def Customize_pulse(self,x):
        with pulse.build(backend=self.backend, default_alignment='sequential', name='Rabi Experiment') as custom_Pulse:
            [pulse.play(self.norm*x, pulse.drive_channel(0))]
            pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        return custom_Pulse
    
    def Customize_pulse_2(self, x, length):
        temp = []
        for i in self.norm:
            for j in range(int(length)):
                temp.append(i)
        temp = np.array(temp)
        with pulse.build(backend=self.backend, default_alignment='sequential', name='Rabi Experiment') as custom_Pulse:
            [pulse.play(temp*x, pulse.drive_channel(0))]
            pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        return custom_Pulse
    
    def draw(self):
        return self.Customize_pulse_2(1,self.length).draw(backend=self.backend)
    
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
    
    def Cali(self,num_rabi_points,length):
        scale_factor = 1e-15
        drive_amp_min = -1
        drive_amp_max = 1
        drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

        rabi_schedules = [self.Customize_pulse_2(a,length) for a in drive_amps]
        #return rabi_schedules
        num_shots_per_point = 1024
        job = self.backend.run(rabi_schedules, 
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
        error_counter=0
        error= [np.inf,np.inf,np.inf]
        while pi_amp > 0.9:
            counter+=1
            if counter==100:
                break
            drive_amps,rabi_values = self.Cali(num_rabi_points,counter)
            if(error[1]==np.inf):
                error_counter=0
                while (error[1]==np.inf):
                    fit_params, y_fit, error = fit_function(drive_amps,
                                         rabi_values, 
                                         lambda x, A, drive_period, phi:(A*np.sin(2*np.pi*x/drive_period - phi)),
                                         [-(np.max(rabi_values)-np.min(rabi_values))/2, test_set[error_counter], 0])
                    #print(error_counter, ": ",test_set[error_counter] , ": " , error)
                    if((error[1] == np.inf) and error_counter<3):
                        error_counter+=1
                    else:
                        break
            else:
                fit_params, y_fit, error = fit_function(drive_amps,
                                         rabi_values, 
                                         lambda x, A, drive_period, phi:(A*np.sin(2*np.pi*x/drive_period - phi)),
                                         [-(np.max(rabi_values)-np.min(rabi_values))/2, test_set[error_counter], 0])
            drive_period = fit_params[1]
            pi_amp = abs(drive_period / 2)
            print(counter, "L: ", pi_amp)
        self.length = counter
        drive_amps,rabi_values = self.Cali(50,counter)
        fit_params, y_fit, error = fit_function(drive_amps,
                                 rabi_values, 
                                 lambda x, A, B, drive_period, phi:(A*np.sin(2*np.pi*x/drive_period - phi) + B),
                                  [-(np.max(rabi_values)-np.min(rabi_values))/2,0, test_set[error_counter], 0])
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
        plt.title(str(self.backend), fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.show()
        return pi_amp,plt
    
    def Cali_l(self,len_max):
        scale_factor = 1e-15
        drive_len = np.linspace(1,len_max,len_max-1)

        rabi_schedules = [self.Customize_pulse_2(1,a) for a in drive_len]
        #return rabi_schedules
        num_shots_per_point = 1024
        job = self.backend.run(rabi_schedules, 
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

def Full_tomography(circ, backend):
    t = time.time()
    test_circ = state_tomography_circuits(circ,[0])
    job = qk.execute(test_circ, backend=backend, shots=8192)
    test_state = StateTomographyFitter(job.result(), test_circ).fit()
    print('Tomography Time taken:', time.time() - t)
    return test_state

def State_Fidel(circ, backend):
    #actual state calculation
    q2 = QuantumRegister(1)
    state = QuantumCircuit(q2)
    state.x(q2[0])
    job = qk.execute(state, backend=Aer.get_backend('statevector_simulator'))
    state_results = job.result().get_statevector(state)
    test_state = Full_tomography(circ,backend)
    t = time.time()
    Fidelity = state_fidelity(state_results,test_state)
    print('Fidelity Fitting Time taken:', time.time() - t)
    return Fidelity
    
def plot_m(state):
    return 0

def our_tomography_circuits(circ,counter):
    cz = qk.QuantumCircuit(1,1,name ="('Z',)")
    cz.compose(circ,[0],inplace= True)
    cz.measure(0,0)
    cy =qk.QuantumCircuit(1,1,name ="('Y',)")
    cy.compose(circ,[0],inplace= True)
    cy.sdg(0)
    cy.h(0)
    cy.measure(0,0)
    cx = qk.QuantumCircuit(1,1,name ="('X',)")
    cx.compose(circ,[0],inplace= True)
    cx.h(0)
    cx.measure(0,0)
    return transpile([cx,cy,cz],backend)

def full_tom(data,backend):
    states_circs= []
    states_pulse= []
    result_states =[]
    DM_states =[]
    temp= [0 for i in range(63)]
    '''qc = qk.QuantumCircuit(1)
    qc.h(0)
    tran_qc= transpile(qc, backend)
    h_pulse_sched= build_schedule(tran_qc,backend)
    qc = qk.QuantumCircuit(1)
    qc.sdg(0)
    tran_qc= transpile(qc, backend)
    sqg_pulse_sched= build_schedule(tran_qc,backend)'''
    #circ = qk.QuantumCircuit(1,1)
    #custom_gate = Gate("temp", 1, [])
    #circ.append(custom_gate, [0])
    counter = 0
    for i in data:
        counter+=1
        circ_c = qk.QuantumCircuit(1,1)
        #print(("temp"+str(counter)))
        custom_gate = Gate("temp"+str(i), 1, [0])
        circ_c.append(custom_gate, [0])
        temp.append(i)
        with pulse.build(backend=backend, default_alignment='sequential', name='anim'+str(counter)) as custom_Pulse:
                pulse.play(temp, pulse.drive_channel(0))
        '''with pulse.build(backend=backend, default_alignment='sequential', name='anim') as custom_Pulse_z:
                pulse.play(temp, pulse.drive_channel(0))
                pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        with pulse.build(backend=backend, default_alignment='sequential', name='anim') as custom_Pulse_y:
                pulse.play(temp, pulse.drive_channel(0))
                pulse.call(sqg_pulse_sched, pulse.drive_channel(0))
                pulse.call(h_pulse_sched, pulse.drive_channel(0))
                pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        with pulse.build(backend=backend, default_alignment='sequential', name='anim') as custom_Pulse_x:
                pulse.play(temp, pulse.drive_channel(0))
                pulse.call(h_pulse_sched, pulse.drive_channel(0))
                pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])'''
        circ_c.add_calibration(custom_gate, [0], custom_Pulse, [counter,1])
        test_circ = our_tomography_circuits(circ_c,[0])
        '''states_pulse.append(custom_Pulse_x)
        states_pulse.append(custom_Pulse_y)
        states_pulse.append(custom_Pulse_z)'''
        states_circs += test_circ
        if(counter < 64):
            del temp[0]
        else:
            if(len(temp)+1%16 == 0):
                continue
            else:
                if(temp.count(0) >= (len(temp)+1)%16):
                    for j in range((len(temp)+1)%16):
                        del temp[0]
                else:
                    [temp.insert(0,0) for i in range(16-(len(temp)+1)%16)]
    #job = backend.run(states_circs, shots=5000)
    #res = job.result()
    job_manager = IBMQJobManager()
    job_set = job_manager.run(states_circs, backend=backend, shots = 3000, name=('Anim'+str(time.strftime("%H:%M:%S", time.localtime()))))
    res = job_set.results().combine_results()
    dict_u = res.to_dict()
    for i in range(len(data)):
        temp_dict = dict_u.copy()
        temp_dict['results'] = temp_dict['results'][3*i:(i+1)*3]
        t_r = res.from_dict(temp_dict)
        #print(states_circs[3*i:(i+1)*3], " -----> ", t_r )
        DM_states.append(StateTomographyFitter(t_r, states_circs[3*i:(i+1)*3]).fit())
        result_states.append([ 2*np.real(DM_states[-1][1][0]),  2*np.imag(DM_states[-1][1][0]), np.real(2*DM_states[-1][0][0]-1)])
        #print(result_states,"   ", DM_states[-1], "\n")
    '''  state.rx(circ[-1],0)
    job = qk.execute(state, backend=Aer.get_backend('statevector_simulator'))
    res = np.array(job.result().get_statevector(state))
    res =  np.tensordot(res,np.transpose(res),0)
    res_state = [ 2*np.real(res[1][0]),  2*np.imag(res[1][0]), np.real(2*res[0][0]-1)]'''
    return result_states, DM_states, states_circs
    
def generate_anim(circ,backend, output):
    plots = []
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    def anim(i):
        ax.clear()
        plot_bloch_vector(plots[i],ax=ax)
        return ax
    plots,DMS,states_pulse = full_tom(circ,backend)
    ani = animation.FuncAnimation(fig, anim, interval=len(circ),frames=len(circ))   
    #ani.save(output, fps=30)
    plt.rcParams["animation.embed_limit"] = 50
    return HTML(ani.to_jshtml())
    plt.show()
    plt.close(fig)
    return states_pulse, DMS

def generate_noise_params(s_pow, w0):
    a = [1]
    NN = 512
    BW = 0.005 # changes narrowness of noise spectrum
    b = si.firwin(NN, BW)*np.cos(w0*np.pi*np.arange(NN))
    b = b/la.norm(b)*np.sqrt(s_pow)
    return a, b

def parametrize_circ(circ,noise_traj_list,backend):
    par = qk.circuit.ParameterVector('thetha', length=int(len(circ)/64))
    batch = []
    for traj in noise_traj_list:
        with pulse.build(backend=backend,default_alignment='sequential') as temp:
            for j in range(int(len(circ)/64)):
                pulse.play(circ[j*64:(j+1)*64], pulse.drive_channel(0))
                pulse.shift_phase(par[j],pulse.drive_channel(0))
            pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        for i in range(len(par)):
            temp.assign_parameters({par[i]: traj[i]}, inplace=True)
        batch.append(block_to_schedule(temp))
    return batch

def shift_all(circ, traj):
    final = []
    start = 0
    for j,i in enumerate(circ):
        final.append((np.cos(start) + 1j*np.sin(start))*i)
        start+=traj[j]
    return final
        
def parametrize_circ_1(circ,noise_traj_list,backend):
    par = qk.circuit.ParameterVector('thetha', length=int(len(circ)))
    batch = []
    for traj in noise_traj_list:
        with pulse.build(backend=backend,default_alignment='sequential') as temp:
            pulse.play(shift_all(circ,traj), pulse.drive_channel(0))
            pulse.measure(qubits=[0], registers=[pulse.MemorySlot(0)])
        batch.append(block_to_schedule(temp))
    return batch

def schwarma_trajectories(a, b, num_gates, num_trajectories):
    # a: coefficients of linear filter numerator
    # b: coefficients of linear filter demoninator
    # num_gates: number of gates in circuit to be "schwarmafied"
    # num_trajectories: number of noise trajectories required
    traj_list = []
    for _ in range(num_trajectories):
        angles = si.lfilter(b, a, np.random.randn(num_gates + 1000))[1000:]
        traj_list.append(angles)
    return traj_list

shots=1000
def runfunc(circ,backend_sim):
    rabi_qobj = transpile(circ, backend=backend_sim)
    results = backend_sim.run(rabi_qobj,shots = shots).result()
    return results


def Spec(data,start,end,num_center_freqs=100,backend=backend, option = 0):  
    #option 0 beaks upp pulse into 64dt chuncks option 1 will morph shift into pulse
    if(option):
        num_gates=int(len(data))
    else:
        num_gates=int(len(data)/64)
    circ_batch = []
    center_idxs=[]
    centers=[]
    all_probs = np.zeros([num_center_freqs, 2])
    for center_idx, center in enumerate(np.linspace(start, end, num_center_freqs)): # vary noise center frequency
        center_idxs.append(center_idx)
        centers.append(center)
        #print('Probing Filter Function at Normalized Frequency: ', center)
        # Generate noise trajectories
        a, b = generate_noise_params(noise_power, center)
        noise_traj_list = np.array(schwarma_trajectories(a, b, num_gates, num_noise_trajs))
    
        # Build noisy circuit dictionary
        if(not option):
            circ_batch+=(parametrize_circ(data,noise_traj_list,backend))
        else:
            circ_batch+=(parametrize_circ_1(data,noise_traj_list,backend))

    # Run circuits
    armonk_model = PulseSystemModel.from_backend(backend)
    backend_sim = PulseSimulator(system_model=armonk_model)
    results = runfunc(circ_batch,backend_sim)#Parallel(n_jobs=cpu_count)(delayed(runfunc)(i,backend_sim) for i in circ_batch)
    '''job_manager = IBMQJobManager()
    job_set = job_manager.run(circ_batch, backend=backend, shots = shots, name=('Spectrosopy'+str(time.strftime("%H:%M:%S", time.localtime()))))
    results = job_set.results()'''
    
    # Compile Results
    cc=0
    prob = 0
    counter = 0
    for i in range(int(len(circ_batch)/num_noise_trajs)):
        for circ in circ_batch[i*num_noise_trajs:((i+1)*num_noise_trajs)]:
            zero_counts = results.get_counts(cc).get('1')
            prob += zero_counts/shots
            cc+=1
        prob = prob/num_noise_trajs
        all_probs[int(center_idxs[counter]), :] = centers[counter], prob
        counter+=1
        prob=0
    
    return spec_data(all_probs, circ_batch)