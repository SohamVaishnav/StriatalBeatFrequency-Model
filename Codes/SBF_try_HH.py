from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import neurodynex3.tools.input_factory as input_factory

"""
Implementation of a Hodging-Huxley neuron
Relevant book chapters:

- http://neuronaldynamics.epfl.ch/online/Ch2.S2.html

"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.


def plot_data(state_monitor, title=None):
## this function has been changed to plot the TimedArray data for the input current 
    """Plots the state_monitor variables ["vm", "I_e", "m", "n", "h"] vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        title (string, optional): plot title to display
    """
    
    print(state_monitor.vm[:].shape)
    plt.subplot(311)
    plt.plot(state_monitor.t / ms, state_monitor.vm[0] / mV, lw=2)

    plt.xlabel("t [ms]")
    plt.ylabel("v [mV]")
    plt.grid()

    plt.subplot(312)

    plt.plot(state_monitor.t / ms, state_monitor.m[0] / volt, "black", lw=2)
    plt.plot(state_monitor.t / ms, state_monitor.n[0] / volt, "blue", lw=2)
    plt.plot(state_monitor.t / ms, state_monitor.h[0] / volt, "red", lw=2)
    plt.xlabel("t (ms)")
    plt.ylabel("act./inact.")
    plt.legend(("m", "n", "h"))
    plt.ylim((0, 1))
    plt.grid()

    plt.subplot(313)
    plt.plot(state_monitor.t / ms, state_monitor.I_e[0] / uamp, lw=2)

    ymin, ymax = np.min(state_monitor.I_e[0] / uamp), np.max(state_monitor.I_e[0] / uamp)

    if ymin == ymax:
        ymin -= 0.1
        ymax += 0.1

    plt.axis([
        0,
        np.max(state_monitor.t / ms),
        ymin * 1.1,
        ymax * 1.1
    ])

    plt.xlabel("t [ms]")
    plt.ylabel("I [micro A]")
    plt.grid()

    if title is not None:
        plt.suptitle(title)

    plt.show()


def simulate_HH_neuron(input_current, simulation_time, num_neurons): 
##function has been changed for our project to take in the number of neurons as an argument

    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]
        num_neurons (int): Number of neurons to simulate

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "m", "n", "h"]
    """

    # neuron parameters
    El = 10.6 * mV
    EK = -12 * mV
    ENa = 115 * mV
    gl = 0.3 * msiemens
    gK = 36 * msiemens
    gNa = 120 * msiemens
    C = 1 * ufarad 

    # forming HH model with differential equations
    eqs = """
    I_e = input_current(t,i) : amp
    membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + \
        gl*(El-vm) + gK*n**4*(EK-vm) : amp
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
    betam = 4*exp(-.0556*vm/mV)/ms : Hz
    betan = .125*exp(-.0125*vm/mV)/ms : Hz
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dvm/dt = membrane_Im/C : volt
    noise : amp
    """

    neuron = NeuronGroup(num_neurons, eqs, threshold= 'vm > 20*mV', method="exponential_euler")

    # parameter initialization
    neuron.vm = 0
    neuron.m = 0.05
    neuron.h = 0.60
    neuron.n = 0.32

    # tracking parameters
    st_mon = StateMonitor(neuron, ["vm", "I_e", "m", "n", "h"], record=True)

    # running the simulation
    hh_net = Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    return st_mon, neuron


## main code
duration_curr = 200
duration_run = 200 * ms
num_neurons = 10

### cortical neuron groups
input_current_single_cortical_g1 = input_factory.get_step_current(5, duration_curr, ms, 12 * uA)
input_current_group_cortical_g1 = TimedArray(
    np.tile(input_current_single_cortical_g1.values, (1, num_neurons)) * amp,
    dt=1 * ms
)
input_current_single_cortical_g2 = input_factory.get_step_current(5, duration_curr, ms, 6.4 * uA)
input_current_group_cortical_g2 = TimedArray(
    np.tile(input_current_single_cortical_g2.values, (1, num_neurons)) * amp,
    dt=1 * ms
)

stat_mon_cortical_g1, cortical_g1 = simulate_HH_neuron(input_current_group_cortical_g1, duration_run, num_neurons)
stat_mon_cortical_g2, cortical_g2 = simulate_HH_neuron(input_current_group_cortical_g2, duration_run, num_neurons)

spike_mon_cortical_g1 = SpikeMonitor(cortical_g1)
spike_mon_cortical_g2 = SpikeMonitor(cortical_g2)

# plot_data(stat_mon_cortical_g1, title="Cortical Neurons (Group 1)")
# plt.show()
# plot_data(stat_mon_cortical_g2, title="Cortical Neurons (Group 2)")
# plt.show()


### striatal neurons
v_threshold_striatal = -55 * mV
v_reset_striatal = -65 * mV
v_rest_striatal = -70 * mV
tau_striatal = 30 * ms
eqs='''
dv/dt = (v_rest_striatal - v + I) / tau_striatal : volt (unless refractory)
I = 0 * volt : volt  # No intrinsic oscillatory input, relies on input from the two cortical neuron groups
'''
neurons_striatal = NeuronGroup(num_neurons, eqs, threshold='v > v_threshold_striatal', 
                               reset='v = v_reset_striatal', method='euler', refractory=5*ms)
neurons_striatal.v = v_rest_striatal

synapses_g1 = Synapses(cortical_g1, neurons_striatal, on_post='v_post += 10 * mV')
synapses_g1.connect(p=1)

synapses_g2 = Synapses(cortical_g2, neurons_striatal, on_post='v_post += 10 * mV')
synapses_g2.connect(p=1)

spike_mon_striatal = SpikeMonitor(neurons_striatal)
state_mon_striatal = StateMonitor(neurons_striatal, 'v', record=True)

neurons_striatal_net = Network(neurons_striatal)
neurons_striatal_net.run(duration_run)

plt.figure(figsize=(12, 8))

plt.subplot(311)
plt.plot(spike_mon_cortical_g1.t / ms, spike_mon_cortical_g1.i, '.r', label='Striatal Neurons')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')
plt.legend()
plt.show()

