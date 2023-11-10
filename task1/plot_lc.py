#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Read data from the file
data = np.genfromtxt('data_lc.txt', delimiter=',', names=True)

# Extract the columns
N = data['N']
MAE_e = data['MAE_e']
MAE_f = data['MAE_f']
RMSE_e = data['RMSE_e']
RMSE_f = data['RMSE_f']

fchl_MAE_e = data['fchl_mae_e']
fchl_MAE_f = data['fchl_mae_f']

MAE_e_slatm = data['MAE_e_slatm']
MAE_f_slatm = data['MAE_f_slatm']
RMSE_e_slatm = data['RMSE_e_slatm']
RMSE_f_slatm = data['RMSE_f_slatm']

MAE_mbdf_e = data['MAE_mbdf_e']
MAE_mbdf_f = data['MAE_mbdf_f']
# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Custom x-axis and y-axis labels for both subplots
custom_xticks = [200, 300, 400, 600, 800, 1000]
custom_yticks_left = [0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4]
custom_yticks_right = [0.9, 1.0, 1.2, 1.4, 1.6, 1.8]

# Plot N vs MAE_e and RMSE_e on the first subplot (left)
axs[0].loglog(N, MAE_e, marker='o', linestyle='-', color='C0', label='QMLighting MAE')
axs[0].loglog(N, fchl_MAE_e, marker='d', linestyle='--', color='C2', label='FCHL19 MAE')
axs[0].loglog(N, RMSE_e, marker='s', linestyle='-', color='C1', label='QMLightning RMSE')
axs[0].loglog(N, MAE_e_slatm, marker='x', linestyle=':', color='C3', label='SLATM MAE')
axs[0].loglog(N, RMSE_e_slatm, marker='v', linestyle=':', color='C4', label='SLATM RMSE')
axs[0].loglog(N[:2], MAE_mbdf_e[:2], marker='^', linestyle='-. ', color='C5', label='MBDF MAE')
axs[0].set_xticks(custom_xticks)
axs[0].set_yticks(custom_yticks_left)
axs[0].set_xticklabels(custom_xticks)
axs[0].set_yticklabels(custom_yticks_left)
axs[0].set_xlabel('N')
axs[0].set_ylabel('Error (kcal/mol)')
# Format the y-axis of the first subplot to disable scientific notation
axs[0].yaxis.set_major_formatter(ScalarFormatter())
# Add a horizontal dashed line at 1 kcal/mol
axs[0].axhline(y=1, color='cyan', linestyle='--')
# Add a legend
axs[0].legend()

# Plot N vs MAE_f and RMSE_f on the second subplot (right)
axs[1].loglog(N, MAE_f, marker='o', linestyle='-', color='C0', label='QMLightning MAE')
axs[1].loglog(N, fchl_MAE_f, marker='d', linestyle='--', color='C2', label='FCHL19 MAE')
axs[1].loglog(N, RMSE_f, marker='s', linestyle='-', color='C1', label='QMLighting RMSE')
axs[1].loglog(N, MAE_f_slatm, marker='x', linestyle=':', color='C3', label='SLATM MAE')
axs[1].loglog(N, RMSE_f_slatm, marker='v', linestyle=':', color='C4', label='SLATM RMSE')
axs[1].loglog(N[:2], MAE_mbdf_f[:2], marker='^', linestyle='-.', color='C5', label='MBDF MAE')
axs[1].set_xticks(custom_xticks)
axs[1].set_yticks(custom_yticks_right)
axs[1].set_xticklabels(custom_xticks)
axs[1].set_yticklabels(custom_yticks_right)
axs[1].set_xlabel('N')
axs[1].set_ylabel('Error (kcal/mol/A)')
# Format the y-axis of the second subplot to disable scientific notation
axs[1].yaxis.set_major_formatter(ScalarFormatter())
# Add a horizontal dashed line at 1 kcal/mol/A
axs[1].axhline(y=1, color='cyan', linestyle='--')
# Add a legend
axs[1].legend()

# Adjust layout to prevent overlap of labels
plt.tight_layout()

# Save the figure as a file (e.g., PNG)
fig.savefig('learning_curves.png')

# Close the figure
plt.close()

