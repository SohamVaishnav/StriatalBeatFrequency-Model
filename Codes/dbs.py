
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

duration = 3 * second 
tau_cortical_1 = 15 * ms
tau_cortical_2 = 15 * ms
tau_striatal = 5 * ms 
v_rest = -70 * mV
v_threshold_corticals = -60*mV
v_threshold_striatal = -55*mV
v_reset = -70 * mV
num_neurons = 10
amp_cort_1 = 25
amp_cort_2 = 30


eqs_cortical_1 = '''
dv/dt = (v_rest - v + I) / tau_cortical_1 : volt (unless refractory)
I = amp_cort_1 * sin(2*pi*freq_cortical_1*t) * mV: volt  # Oscillatory input
freq_cortical_1 : Hz
'''

eqs_cortical_2 = '''
dv/dt = (v_rest - v + I) / tau_cortical_2 : volt (unless refractory)
I = amp_cort_2 * sin(2*pi*freq_cortical_2*t) * mV: volt  # Oscillatory input
freq_cortical_2 : Hz
'''

eqs_striatal = '''
dv/dt = (v_rest - v + I + DBS_input) / tau_striatal : volt (unless refractory)
I : volt  # Synaptic input from cortical neurons
DBS_input = 25 * mV * sin(2 * pi * 130 * Hz * t) : volt  # High-frequency DBS
'''

neurons_cortical_1 = NeuronGroup(num_neurons, eqs_cortical_1, threshold='v > v_threshold_corticals',
                       reset='v = v_reset', method='euler', refractory=1*ms)
neurons_cortical_2 = NeuronGroup(num_neurons, eqs_cortical_2, threshold='v > v_threshold_corticals',
                       reset='v = v_reset', method='euler', refractory=1*ms)

neurons_striatal = NeuronGroup(5, eqs_striatal, threshold='v > v_threshold_striatal',
                          reset='v = v_reset', method='euler', refractory=5*ms)

neurons_cortical_1.v = v_rest
neurons_cortical_2.v = v_rest
neurons_striatal.v = v_rest
neurons_cortical_1.freq_cortical_1 = 14 * Hz #beta wave
neurons_cortical_2.freq_cortical_2 = 24 * Hz #beta wave

synapses_cort1_striat = Synapses(neurons_cortical_1, neurons_striatal, on_pre='v_post += 1.3 * mV') 
synapses_cort1_striat.connect(p=0.7)
synapses_cort2_striat = Synapses(neurons_cortical_2, neurons_striatal, on_pre='v_post += 1.3 * mV')
synapses_cort2_striat.connect(p=0.7)

spikemon_cortical_1 = SpikeMonitor(neurons_cortical_1)
spikemon_cortical_2 = SpikeMonitor(neurons_cortical_2)
statemon_cortical_1 = StateMonitor(neurons_cortical_1, 'v', record=True)
statemon_cortical_2 = StateMonitor(neurons_cortical_2, 'v', record=True)

spikemon_striatal = SpikeMonitor(neurons_striatal)
statemon_striatal = StateMonitor(neurons_striatal, 'v', record=True)

run(duration)

figure(figsize=(12, 8))

subplot(211)
plot(spikemon_cortical_1.t / ms, spikemon_cortical_1.i + num_neurons, '.k', label='Group 1')
plot(spikemon_cortical_2.t / ms, spikemon_cortical_2.i + 2*num_neurons, '.r', label='Group 2') 
plot(spikemon_striatal.t / ms, spikemon_striatal.i, '.b', label='Striatal')
xlabel('Time (ms)')
ylabel('Neuron index')
title('Spike Raster Plot')
legend()

subplot(212)
for i in range(num_neurons):
    plot(statemon_cortical_1.t / ms, statemon_cortical_1.v[i] / mV, color='black', alpha=0.1, label='Group 1')
for i in range(num_neurons):
    plot(statemon_cortical_2.t / ms, statemon_cortical_2.v[i] / mV, color='red', alpha=0.1, label='Group 2') 
for i in range(5):
    plot(statemon_striatal.t / ms, statemon_striatal.v[i] / mV, color='blue', alpha=0.1, label='Striatal')
xlabel('Time (ms)')
ylabel('Membrane potential (mV)')
title('Membrane Potential Over Time')
plt.tight_layout()
plt.show()
