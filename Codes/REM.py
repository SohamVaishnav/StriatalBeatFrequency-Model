from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

duration = 5 * second 
cortical_tau = 15 * ms
cortical_v_rest = -70 * mV
cortical_v_threshold = -62 * mV
cortical_v_reset = -65 * mV
cortical_noise_amplitude = 0 * mV

striatal_tau = 30 * ms
striatal_v_rest = -70 * mV
striatal_v_threshold = -60 * mV
striatal_v_reset = -65 * mV
striatal_noise_amplitude = 0 * mV

eqs_cortical_1 = '''
dv/dt = (cortical_v_rest - v + I + noise) / cortical_tau : volt (unless refractory)
I = 20*mV * sin(2 * pi * freq * t): volt  # Oscillatory input
freq : Hz  # Frequency of oscillation
noise : volt  # Random fluctuations (noise)
'''
eqs_cortical_2 = '''
dv/dt = (cortical_v_rest - v + I + noise) / cortical_tau : volt (unless refractory)
I = 15*mV * sin(2 * pi * freq * t): volt  # Oscillatory input
freq : Hz  # Frequency of oscillation
noise : volt  # Random fluctuations (noise)
'''
eqs_striatal = '''
dv/dt = (striatal_v_rest - v + I + noise) / striatal_tau : volt (unless refractory)
input_sum : volt  # Combined input from cortical groups
noise : volt  # Random fluctuations (noise)
I = 0 * volt : volt 
'''

group1 = NeuronGroup(10, eqs_cortical_1, threshold='v > cortical_v_threshold',
                     reset='v = cortical_v_reset', method='euler', refractory=10*ms)
group2 = NeuronGroup(10, eqs_cortical_2, threshold='v > cortical_v_threshold',
                     reset='v = cortical_v_reset', method='euler', refractory=10*ms)
group3 = NeuronGroup(10, eqs_striatal, threshold='v > striatal_v_threshold',
                     reset='v = striatal_v_reset', method='euler', refractory=10*ms)

group1.v = cortical_v_rest
group1.noise = cortical_noise_amplitude * randn(len(group1))
group1.freq = 20 * Hz  #Beta wave

group2.v = cortical_v_rest
group2.noise = cortical_noise_amplitude * randn(len(group2))
group2.freq = 6 * Hz  #Theta wave

group3.v = striatal_v_rest
group3.noise = striatal_noise_amplitude * randn(len(group3))

syn1 = Synapses(group1, group3, on_pre='v_post += 0.55 * mV')
syn2 = Synapses(group2, group3, on_pre='v_post += 0.55 * mV')
syn1.connect(p=0.5)
syn2.connect(p=0.5)

spike_monitor1 = SpikeMonitor(group1)
spike_monitor2 = SpikeMonitor(group2)
spike_monitor3 = SpikeMonitor(group3)
state_monitor1 = StateMonitor(group1, 'v', record=True)
state_monitor2 = StateMonitor(group2, 'v', record=True)
state_monitor3 = StateMonitor(group3, 'v', record=True)

run(duration)

plt.figure(figsize=(12, 12))
plt.subplot(311)
plt.plot(spike_monitor1.t / ms, spike_monitor1.i, '.b', label='Cortical (Group 1: Beta)')
plt.plot(spike_monitor2.t / ms, spike_monitor2.i + 50, '.y', label='Cortical (Group 2: Theta)')
plt.plot(spike_monitor3.t / ms, spike_monitor3.i + 100, '.r', label='Striatal (Group 3)')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')
plt.legend()

plt.subplot(312)
for i in range(10):
    plt.plot(state_monitor1.t / ms, state_monitor1.v[i] / mV, color='blue', alpha=0.3, label='Group 1' if i == 0 else "")
    plt.plot(state_monitor2.t / ms, state_monitor2.v[i] / mV, color='yellow', alpha=0.3, label='Group 2' if i == 0 else "")
    plt.plot(state_monitor3.t / ms, state_monitor3.v[i] / mV, color='red', alpha=0.3, label='Group 3' if i == 0 else "")
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Membrane Potential Over Time')
plt.legend()

time = np.arange(0, float(duration), 0.001)
input1 = 15 * np.sin(2 * np.pi * 20 * time)
input2 = 15 * np.sin(2 * np.pi * 6 * time)
plt.subplot(313)
plt.plot(time, input1, label='Beta Waves (20 Hz)')
plt.plot(time, input2, label='Theta Waves (6 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Input (mV)')
plt.title('Oscillatory Inputs')
plt.legend()

plt.tight_layout()
plt.show()