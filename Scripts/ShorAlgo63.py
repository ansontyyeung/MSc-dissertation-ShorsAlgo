#!/usr/bin/env python
# coding: utf-8

# In[1]:


#initialization
import numpy as np
import math
import pandas as pd

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import qiskit_aer.noise as noise
from qiskit.providers.aer.noise.errors import *  
from qiskit_aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
pd.set_option('display.max_rows', None)

# import os for save fig to local machine
import os
import io
import sys
from contextlib import redirect_stdout

# setup your own MY_API_TOKEN to use real QPU backend
# from qiskit import IBMQ
# MY_API_TOKEN = '8bed81dc8c0e24350ef14c720816ea55938c5ce7a83deff435c3720f0c33eff1df409a6dba874763473a00f90cdb2b4dad20881000e930d1097235f582f411a5'
# IBMQ.save_account(MY_API_TOKEN, overwrite=True)
print("Imports Successful")

# In[2]:


# A function to fix cannot show plots in notebooks when generated inside of loops
# Solution: close the figure instance before returning it, otherwise it would be rendered in the notebook
def show_figure(fig):
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)


# In[3]:


def process_simulation_results(key, list_rows, list_keys, list_correct_percentage):
    
    # Print as a table
    headers=["Phase", "Fraction", "Guess for r", "Count"]
    df = pd.DataFrame(list_rows, columns=headers)
    print(df)
    
    # Filter the DataFrame based on the additional criteria
    filtered_df = df[(df['Guess for r'] % 2 == 0) &
                     (df['Fraction'].apply(lambda x: math.gcd(int(x.split('/')[0]), int(x.split('/')[1])) == 1)) &
                     (df['Fraction'].apply(lambda x: (pow(a, int(x.split('/')[1])/2))%N != 1))]

    # Get the "Guess for r" column data from the filtered DataFrame
    guess_for_r_df = filtered_df[['Guess for r', 'Count']]

    # Generate the guesses for each "Guess for r" value
    guesses = []
    for r, count in zip(guess_for_r_df['Guess for r'],guess_for_r_df['Count']):
        guess1 = math.gcd(a**(r//2)-1, N)
        guess2 = math.gcd(a**(r//2)+1, N)
        guesses.append([r, guess1, guess2, count])

    # Create a DataFrame from the guesses
    result_df = pd.DataFrame(guesses, columns=['Guess for r', 'Guess 1', 'Guess 2', 'Count'])

    # Sort the DataFrame by the "Guess for r" column
    result_df = result_df.sort_values('Guess for r', key=lambda x: x.map(lambda y: result_df['Guess for r'].value_counts()[y]), ascending=False)
    
    # Sum the 'Count' column based on the 'Guess for r' column
    sum_by_guess = result_df.groupby('Guess for r')['Count'].sum().to_dict()

    # Add the 'Sum by Guess' column to the original DataFrame
    result_df['Sum by Guess'] = result_df['Guess for r'].map(sum_by_guess)

    # Add a column for the repetition count
    result_df['Repetition Count'] = result_df['Guess for r'].map(result_df['Guess for r'].value_counts())
    
    # Check which rows have Guess1 * Guess2 = N
    try:
        correct_count = result_df.loc[(result_df['Guess 1'] * result_df['Guess 2'] == N) & 
                                      (result_df['Guess 1'] != N) & (result_df['Guess 2'] != N), 'Sum by Guess'].iloc[0]
    except IndexError:
        # Handle the case where the DataFrame is empty
        correct_count = 0

    # Get the total number of repetitions
    total_count = result_df['Count'].sum()
    

    # Calculate the percentage of correct results
    correct_percentage = (correct_count / total_count) * 100

    # Print the unique rows
    result_df = result_df[[col for col in result_df.columns if col != 'Count']]
    result_df = result_df.drop_duplicates()
    print(result_df)
    
    return correct_count, correct_percentage, total_count


# In[4]:


def analyze_simulation_results_noise(simulation_results, is_thermal=False):
    """
    Parameters:
    simulation_results (dict): A dictionary where the keys are the number of shots and the values are another dictionary with binary strings as keys and their counts as values.

    Returns:
    dict: A dictionary containing the analysis results, including the correct result count and percentage.
    """
    list_keys, list_correct_percentage = [], [] # For plotting success rate comparison
    list_keys_T1, list_keys_T2 = [], [] # For thermal case
    
    for key, counts in simulation_results.items():
        list_rows, list_measured_phases, list_count = [], [], []
        for binary, count in counts.items():
            decimal = int(binary, 2)  # Convert (base 2) string to decimal
            phase = decimal/(2**N_COUNT)  # Find corresponding eigenvalue
            list_measured_phases.append(phase)
            list_count.append(count)
   
        for phase, count in zip(list_measured_phases, list_count):
            frac = Fraction(phase).limit_denominator(N)
            list_rows.append([phase, f"{frac.numerator}/{frac.denominator}", frac.denominator, count])
             
        correct_count, correct_percentage, total_count = process_simulation_results(key, list_rows, list_keys, list_correct_percentage)
        list_correct_percentage.append(correct_percentage)
        
        # Append success rate to new DF
        if is_thermal==False:
            list_keys.append(f"{float(key.split('=')[1])}")
        else:
            key_parts = key.split(';')
            T1_value = float(key_parts[0].split('=')[1])
            T2_value = float(key_parts[1].split('=')[1])
            list_keys_T1.append(f"{T1_value / 1000:.2f}")
            list_keys_T2.append(f"{T2_value / 1000:.2f}")
  
    # Print the result
    print(f"Correct result: {correct_count} out of {total_count} ({correct_percentage:.2f}% correct)")
    
    if is_thermal==False:
        success_comparison_df = pd.DataFrame({
            'Key': list_keys,
            'Correct percentage': list_correct_percentage
        })
    else:
        success_comparison_df = pd.DataFrame({
            'Key_T1': list_keys_T1,
            'Key_T2': list_keys_T2,
            'Correct percentage': list_correct_percentage
        })
        
    print(success_comparison_df)
        
    return success_comparison_df


# In[5]:


def store_file(file_path, counts, is_thermal=False):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        with redirect_stdout(io.StringIO()):
            analyze_simulation_results_noise(counts, is_thermal)
            stdout_output = sys.stdout.getvalue()
            file.write(stdout_output)


# In[6]:


N = 63


# In[7]:


#defining the U gates for mod 63
def c_amod63(a, power):
    """Controlled multiplication by a mod 63"""
    if a not in [2, 4, 5, 8, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 26, 29, 31, 32, 34, 37, 38, 40, 41, 43, 44, 46, 47, 50, 52, 53, 55, 58, 59, 62]:
        raise ValueError("'a' must be 2, 4, 5, 8, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 26, 29, 31, 32, 34, 37, 38, 40, 41, 43, 44, 46, 47, 50, 52, 53, 55, 58, 59, or 62")
    #the number of qubits used is 6 
    U = QuantumCircuit(6) 
    #implementing the swap gates for rotation 
    #we implement every number and see common behaivor between the numbers 
    for iteration in range(power):
        if a in [2,61]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
            U.swap(3,4)
            U.swap(4,5)
    U = U.to_gate()
    U.name = f"{a}^{power} mod 63"
    c_U = U.control()
    return c_U


# In[8]:


# Specify variables
#N_COUNT = 2*N.bit_length()+3-6 #number of counting qubits
N_COUNT = 9
print(N_COUNT)
a = 2 #13,23,29,31,
# a=40; 341 sec
# a=2; 14sec


# In[9]:


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc


# In[10]:


# Create QuantumCircuit with n_count counting qubits
# plus 10 qubits for U to act on
qc = QuantumCircuit(N_COUNT+6, N_COUNT)

# Initialize counting qubits
# in state |+>
for q in range(N_COUNT):
    qc.h(q)
    
# And auxiliary register in state |1>
qc.x(N_COUNT+6)

# Do controlled-U operations
for q in range(N_COUNT):
    qc.append(c_amod63(a, 2**q), 
             [q] + [i+N_COUNT for i in range(6)])

# Do inverse-QFT
qc.append(qft_dagger(N_COUNT), range(N_COUNT))

# Measure circuit
qc.measure(range(N_COUNT), range(N_COUNT))
qc.draw(fold=-1)  # -1 means 'do not fold' 


# In[11]:


fac63_noise_free_counts={}
# Create empty noise simulator backend
sim_noise = AerSimulator(noise_model=NoiseModel())
Transpile circuit for basis gates
circ_tnoise = transpile(qc, sim_noise)

# Run and get counts
noise_free_result = sim_noise.run(circ_tnoise, shots=1024).result()
print("Time taken to run noise free: {} sec".format(noise_free_result.time_taken))
noise_free_counts = noise_free_result.get_counts(0)
# Sort the result
sorted_noise_free_counts = dict(sorted(noise_free_counts.items(), key=lambda x: int(x[0])))
key = f"error=0"
fac63_noise_free_counts[key] = sorted_noise_free_counts

# Plot empty noise output
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_histogram(sorted_noise_free_counts, ax=ax)
ax.grid(False)
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_noise-free.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[12]:


fac63_noise_free_success_comparison_df=analyze_simulation_results_noise(fac63_noise_free_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_noise-free_data.txt'
store_file(file_path, fac63_noise_free_counts)


# In[13]:


all_fac63_noise_free_counts = {}
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
# Create empty noise simulator backend
sim_noise = AerSimulator(noise_model=NoiseModel())
# Transpile circuit for basis gates
circ_tnoise = transpile(qc, sim_noise)

# Run with different shot values and store the results
shot_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
fac63_noise_free_results = {}

for i, shots in enumerate(shot_values):
    result_shots = sim_noise.run(circ_tnoise, shots=shots).result()
    count_shots = result_shots.get_counts(0)
    sorted_counts_shots = dict(sorted(count_shots.items(), key=lambda x: int(x[0])))
    
    plot_histogram(sorted_counts_shots, ax=axes[i])
    axes[i].grid(False)  # No idea why last plot shows horizontal grid line, add this code to prevent
    axes[i].set_title(f"Shots={shots}")
    
    # Store the results
    key = f"Shots={shots}"
    all_fac63_noise_free_counts[key] = sorted_counts_shots
    
    print("Time taken for {} shots: {} sec".format(key, result_shots.time_taken))

plt.suptitle("Factor 63: Noise-free Simulation Results with Varying Shot Values")
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_noise-free_vary_shots.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[14]:


fac63_noise_free_shots_success_comparison_df = analyze_simulation_results_noise(all_fac63_noise_free_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_noise-free_vary_shots_data.txt'
store_file(file_path, all_fac63_noise_free_counts)


# In[15]:


# Create the line graph
plt.figure(figsize=(6, 4.5))
plt.plot(fac63_noise_free_shots_success_comparison_df['Key'], fac63_noise_free_shots_success_comparison_df['Correct percentage'],marker='o', markersize=5)
plt.xlabel('Shots')
plt.ylabel('Success Rate')
plt.title('Factor 63 Success Rate with Varying Shots under Noise-Free Condition')
#plt.gca().invert_yaxis()
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))

plt.grid(True)
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_noise-free_vary_shots_success_rate.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[16]:


qr = QuantumRegister(6)
meas_calibs, state_labels = qiskit.utils.mitigation.circuits.complete_meas_cal(qubit_list=[0,1,2,3,4,5], qr=qr)

# Step through the values of qubitk_p0_given1 and qubitk_p1_given0
start = 1e-3
end = 4.93e-1
num_steps = 6
all_fac63_ro_noise_counts = {}
figures = []  # Create a list to store the figures
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

for i in range(num_steps):
    qubitk_p0_given1 = start + i * (end - start) / (num_steps - 1)
    qubitk_p1_given0 = start + i * (end - start) / (num_steps - 1)
    qubitk_p0_given0 = 1 - qubitk_p0_given1
    qubitk_p1_given1 = 1 - qubitk_p1_given0

    # Add errors to noise model
    noise_ro = NoiseModel()
    for qi in range(6):
        read_err = noise.errors.readout_error.ReadoutError([[qubitk_p0_given0, qubitk_p0_given1],
                                                            [qubitk_p1_given0, qubitk_p1_given1]])
        noise_ro.add_readout_error(read_err, [qi])

    # Run the noisy simulation
    sim_readout = AerSimulator(noise_model=noise_ro)
    circ_readout = transpile(qc, sim_readout)
    result_readout = sim_readout.run(circ_readout, shots=1024).result()
    count_readout = result_readout.get_counts(0)
    sorted_counts_readout = dict(sorted(count_readout.items(), key=lambda x: int(x[0])))
    
    plot_histogram(sorted_counts_readout, ax=axes[i])
    axes[i].grid(False)  # No idea why last plot shows horizontal grid line, add this code to prevent
    axes[i].set_title(f"Readout Error={qubitk_p1_given0:.3f}")

    # Store the results
    key = f"readout_error={qubitk_p1_given0:.3e}"
    all_fac63_ro_noise_counts[key] = sorted_counts_readout
    
    print("Time taken to run readout error={}: {} sec".format(key,result_readout.time_taken))
    
plt.suptitle("Factor 63: Readout Noise Simulation Results with Varying Shot Values")
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_readout.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[17]:


fac63_ro_noise_success_comparison_df=analyze_simulation_results_noise(all_fac63_ro_noise_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_readout_data.txt'
store_file(file_path, all_fac63_ro_noise_counts)


# In[18]:


plt.figure(figsize=(6, 4.5))
plt.plot(list(range(len(fac63_ro_noise_success_comparison_df['Key']))), fac63_ro_noise_success_comparison_df['Correct percentage'],marker='o', markersize=5, label='shots=1024')
plt.xticks(list(range(len(fac63_ro_noise_success_comparison_df['Key']))), fac63_ro_noise_success_comparison_df['Key'])
plt.xlabel('Readout Error')
plt.ylabel('Success Rate')
plt.title('Factor 63 Success Rate with Readout Error')
#plt.gca().invert_yaxis()
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
plt.grid(True)
plt.legend()
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_readout_success_rate.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[17]:


# we assume the same depolarizing error for each
start = 10e-6
end = 10e-4
num_steps = 6
all_fac63_depolar_noise_counts = {}
figures = []  # Create a list to store the figures
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

for i in range(num_steps):
    d_error = start + i * (end - start) / (num_steps - 1)

    # Add errors to noise model
    noise_depolar = NoiseModel()
    noise_depolar.add_all_qubit_quantum_error(depolarizing_error(d_error*4/3, 1), ['h','x','measure']) #4^n*P/(4^n-1)
    noise_depolar.add_all_qubit_quantum_error(depolarizing_error(d_error*16/15, 2), ['swap','cp'])
        
    # Run the noisy simulation
    sim_depolar = AerSimulator(noise_model=noise_depolar)
    circ_depolar = transpile(qc, sim_depolar)
    result_depolar = sim_depolar.run(circ_depolar, shots=1024).result()
    count_depolar = result_depolar.get_counts(0)
    sorted_counts_depolar = dict(sorted(count_depolar.items(), key=lambda x: int(x[0])))
    
    plot_histogram(sorted_counts_depolar, ax=axes[i])
    axes[i].grid(False)  # No idea why last plot shows horizontal grid line, add this code to prevent
    axes[i].set_title(f"Depolarising Error={d_error:.3f}")

    # Store the results
    key = f"depolar_error={d_error:.3e}"
    all_fac63_depolar_noise_counts[key] = sorted_counts_depolar
    
    print("Time taken to run depolarising error={}: {} sec".format(key,result_depolar.time_taken)) 
    
plt.suptitle("Factor 63: Depolarising Error Simulation Results")
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_depolarising_small.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[18]:


fac63_dp_noise_success_comparison_df=analyze_simulation_results_noise(all_fac63_depolar_noise_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_depolarising_data_small.txt'
store_file(file_path, all_fac63_depolar_noise_counts)


# In[19]:


plt.figure(figsize=(6, 4.5))
plt.plot(fac63_dp_noise_success_comparison_df['Key'], fac63_dp_noise_success_comparison_df['Correct percentage'],marker='o', markersize=5, label='shots=1024')
plt.xlabel('Depolarising Error')
plt.ylabel('Success Rate')
plt.title('Factor 63 Success Rate with Depolarising Error')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
plt.grid(True)
plt.legend()
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_depolarising_success_rate_small.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[22]:


start_T1s = 0.78e3
end_T1s = 629.55e3
start_T2s = 0.21e3
end_T2s = 578.24e3
# Instruction times (in nanoseconds)
time_one_qubit = 70
time_two_qubit = 559
time_measure = 70
num_steps = 6
all_fac63_thermal_noise_counts = {}
figures = []  # Create a list to store the figures
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

for i in range(num_steps):
    T1 = end_T1s - i * (end_T1s - start_T1s) / (num_steps - 1)
    T2 = end_T2s - i * (end_T2s - start_T2s) / (num_steps - 1)

    # QuantumError
    errors_measure = thermal_relaxation_error(T1, T2, time_measure)
    errors_one_qubit  = thermal_relaxation_error(T1, T2, time_one_qubit)
    errors_two_qubit = thermal_relaxation_error(T1, T2, time_two_qubit).expand(thermal_relaxation_error(T1, T2, time_two_qubit))
#     print("error one qubit rate is: {}".format(errors_one_qubit.to_dict()))
#     print("error two qubit rate is: {}".format(errors_two_qubit.to_dict()))
    
    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(15):
        noise_thermal.add_quantum_error(errors_measure, "measure", [j])
        noise_thermal.add_quantum_error(errors_one_qubit, ["h","x"], [j])
        for k in range(15):
            noise_thermal.add_quantum_error(errors_two_qubit, ["cp","swap"], [j, k])
            
    # Run the noisy simulation
    sim_thermal = AerSimulator(noise_model=noise_thermal)
    circ_thermal = transpile(qc, sim_thermal)
    result_thermal = sim_thermal.run(circ_thermal, shots=1024).result()
    counts_thermal = result_thermal.get_counts(0)
    sorted_counts_thermal = dict(sorted(counts_thermal.items(), key=lambda x: int(x[0])))
    
    plot_histogram(sorted_counts_thermal, ax=axes[i])
    axes[i].grid(False)  # No idea why last plot shows horizontal grid line, add this code to prevent
    axes[i].set_title(f"T1={T1:.3f} T2={T2:.3f}")

    # Store the results
    key = f"T1={T1};T2={T2}"
    all_fac63_thermal_noise_counts[key] = sorted_counts_thermal
    
    print("Time taken to run thermal relaxation error={}: {} sec".format(key,result_thermal.time_taken)) 
    
plt.suptitle("Factor 63: Thermal Relaxation Error Simulation Results")
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_thermal.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[29]:


fac63_thermal_noise_success_comparison_df=analyze_simulation_results_noise(all_fac63_thermal_noise_counts, is_thermal=True)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_thermal_data.txt'
store_file(file_path, all_fac63_thermal_noise_counts, is_thermal=True)


# In[30]:


plt.figure(figsize=(10, 4.5))
plt.plot(fac63_thermal_noise_success_comparison_df['Key_T1'], fac63_thermal_noise_success_comparison_df['Correct percentage'],marker='o', markersize=5, label='shots=1024')
plt.xlabel('Thermal Relaxation Error')
plt.ylabel('Success Rate')
plt.title('Factor 63 Success Rate with Thermal Relaxation Error')
x_labels = ["T1={}us\nT2={}us".format(t1,t2) for t1, t2 in zip(fac63_thermal_noise_success_comparison_df['Key_T1'], fac63_thermal_noise_success_comparison_df['Key_T2'])]
plt.xticks(fac63_thermal_noise_success_comparison_df['Key_T1'], x_labels)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
plt.grid(True)
plt.legend()
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_thermal_success_rate.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[34]:


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

median_readout=0.015 
median_depolarising = 0.0033
median_T1=300e3
median_T2=135e3

# Noise-Free
fac63_noise_free_counts={}
sim_noise = AerSimulator(noise_model=NoiseModel())
circ_tnoise = transpile(qc, sim_noise)
noise_free_result = sim_noise.run(circ_tnoise, shots=1024).result()
noise_free_counts = noise_free_result.get_counts(0)
# Sort the result
sorted_noise_free_counts = dict(sorted(noise_free_counts.items(), key=lambda x: int(x[0])))
plot_histogram(sorted_noise_free_counts, ax=axes[0])
axes[0].set_title("Noise-Free")
key = f"error=0"
fac63_noise_free_counts[key] = sorted_noise_free_counts

# Readout Median Sim
noise_ro = NoiseModel()
fac63_ro_noise_counts = {}
qubitk_p0_given1 = median_readout
qubitk_p1_given0 = median_readout
qubitk_p0_given0 = 1 - qubitk_p0_given1
qubitk_p1_given1 = 1 - qubitk_p1_given0
for qi in range(6):
read_err = noise.errors.readout_error.ReadoutError([[qubitk_p0_given0, qubitk_p0_given1],
                                                    [qubitk_p1_given0, qubitk_p1_given1]])
noise_ro.add_readout_error(read_err, [qi])
sim_readout = AerSimulator(noise_model=noise_ro)
circ_readout = transpile(qc, sim_readout)
result_readout = sim_readout.run(circ_readout, shots=1024).result()
count_readout = result_readout.get_counts(0)
sorted_counts_readout = dict(sorted(count_readout.items(), key=lambda x: int(x[0])))
plot_histogram(sorted_counts_readout, ax=axes[1])
axes[1].set_title(f"Readout Error={median_readout:.3f}")
key = f"readout_error={median_readout:.3e}"
fac63_ro_noise_counts[key] = sorted_counts_readout

# Depolarising Median Sim
noise_depolar = NoiseModel()
fac63_depolar_noise_counts = {}
noise_depolar.add_all_qubit_quantum_error(depolarizing_error(median_depolarising*4/3, 1), ['h','x','measure']) #4^n*P/(4^n-1)
noise_depolar.add_all_qubit_quantum_error(depolarizing_error(median_depolarising*16/15, 2), ['swap','cp'])
sim_depolar = AerSimulator(noise_model=noise_depolar)
circ_depolar = transpile(qc, sim_depolar)
result_depolar = sim_depolar.run(circ_depolar, shots=1024).result()
count_depolar = result_depolar.get_counts(0)
sorted_counts_depolar = dict(sorted(count_depolar.items(), key=lambda x: int(x[0])))
plot_histogram(sorted_counts_depolar, ax=axes[2])
axes[2].set_title(f"Depolarising Error={median_depolarising:.4f}")
key = f"depolar_error={median_depolarising:.4e}"
fac63_depolar_noise_counts[key] = sorted_counts_depolar

# Thermal Relaxation Median Sim
noise_thermal = NoiseModel()
fac63_thermal_noise_counts = {}
time_one_qubit = 70
time_two_qubit = 559
time_measure = 70
errors_measure = thermal_relaxation_error(median_T1, median_T2, time_measure)
errors_one_qubit  = thermal_relaxation_error(median_T1, median_T2, time_one_qubit)
errors_two_qubit = thermal_relaxation_error(median_T1, median_T2, time_two_qubit).expand(thermal_relaxation_error(median_T1, median_T2, time_two_qubit))
print("one qubit error rate is: {}".format(errors_one_qubit.to_dict()))
print("two qubit error rate is: {}".format(errors_two_qubit.to_dict()))
for j in range(15):
noise_thermal.add_quantum_error(errors_measure, "measure", [j])
noise_thermal.add_quantum_error(errors_one_qubit, ["h","x"], [j])
for k in range(15):
    noise_thermal.add_quantum_error(errors_two_qubit, ["cp","swap"], [j, k])
sim_thermal = AerSimulator(noise_model=noise_thermal)
circ_thermal = transpile(qc, sim_thermal)
result_thermal = sim_thermal.run(circ_thermal, shots=1024).result()
counts_thermal = result_thermal.get_counts(0)
sorted_counts_thermal = dict(sorted(counts_thermal.items(), key=lambda x: int(x[0])))
plot_histogram(sorted_counts_thermal, ax=axes[3])
axes[3].grid(False)  # No idea why last plot shows horizontal grid line, add this code to prevent
axes[3].set_title(f"Thermal Relaxation Error: T1={median_T1:.3f} T2={median_T2:.3f}")
key = f"T1={median_T1};T2={median_T2}"
fac63_thermal_noise_counts[key] = sorted_counts_thermal

plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_median.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()


# In[35]:


fac63_noise_free_success_comparison_df = analyze_simulation_results_noise(fac63_noise_free_counts)
fac63_median_readout_noise_success_comparison_df = analyze_simulation_results_noise(fac63_ro_noise_counts)
fac63_median_depolar_noise_success_comparison_df = analyze_simulation_results_noise(fac63_depolar_noise_counts)
fac63_median_thermal_noise_success_comparison_df = analyze_simulation_results_noise(fac63_thermal_noise_counts, is_thermal=True)

file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_median_data_free.txt'
store_file(file_path, fac63_noise_free_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_median_data_readout.txt'
store_file(file_path, fac63_ro_noise_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_median_data_depolarising.txt'
store_file(file_path, fac63_depolar_noise_counts)
file_path = r'/user/home/mh23476/MSc_Dissertation/SimResults/Data/fac63_median_data_thermal.txt'
store_file(file_path, fac63_thermal_noise_counts, is_thermal=True)


# In[36]:


df = pd.concat([fac63_noise_free_success_comparison_df,
            fac63_median_readout_noise_success_comparison_df,
            fac63_median_depolar_noise_success_comparison_df,
            fac63_median_thermal_noise_success_comparison_df])

# Create the plot
plt.figure(figsize=(6, 4.5))
df.plot(kind='bar', legend=False,  rot=0)
plt.legend(['shots=1024'])
plt.ylabel('Correct Percentage (%)')
plt.xticks(range(len(df)), ['Noise Free', 'Readout\nError', 'Depolarising\nError', 'Thermal Relaxation\nError'])

# Set the plot title
plt.title('Success Rate under Noise-Free and Median Error Conditions')
plt.tight_layout()
save_path = r"/user/home/mh23476/MSc_Dissertation/SimResults/Plot/fac63_median_success_rate.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()