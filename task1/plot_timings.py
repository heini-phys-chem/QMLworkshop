#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Read data from fchl_timings.txt
fchl_data = np.genfromtxt('fchl_timings.txt', delimiter=',', names=True, dtype=None)

# Read data from qmliggnting_timings.txt
qml_data = np.genfromtxt('qmliggnting_timings.txt', delimiter=',', names=True, dtype=None)

# Extract the relevant columns
N_fchl = fchl_data['N']
t_rep = fchl_data['t_rep']  # Convert to seconds
t_kernel = fchl_data['t_kernel']  # Convert to seconds
t_alpha = fchl_data['t_alpha']

N_qml = qml_data['N']
t_train_min = qml_data['t_train_min']  # Convert to seconds

# Create the bar plot with thicker bars
fig, ax1 = plt.subplots()

# Adjust the width for thicker bars
width = 50

ax1.bar(N_fchl - width, t_kernel, width, label='FCHL19 kernel', alpha=0.7)
ax1.bar(N_fchl, t_alpha, width, label='FCHL19 alpha', alpha=0.7)
ax1.bar(N_fchl + width, t_train_min, width, label='QMLightning training', alpha=0.7)

# Set labels and legends
ax1.set_xlabel('N')
ax1.set_ylabel('Time (min)')  # Update the label to seconds
ax1.legend(loc='upper left', fontsize='small')

plt.title('Timing Comparison')

# Save the figure as a PNG file
plt.savefig('timing_comparison.png', dpi=300, bbox_inches='tight')

# Close the figure to release resources
plt.close()

