# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:20:46 2024

@author: spide
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the linear equation x2 = -1.265782808 * x1 + 147.4056772 (j=4)



def calculate_x2(x1):
    return -1.265782808 * x1 + 147.4056772

def calculate_x2_j1(x1):
    return -1.265782808 * x1 + 176.6759548

def calculate_x2_j3(x1):
    return -1.265782808 * x1 + 354.4410007

# Generate a range of x1 values
x1_values = np.linspace(0, 300, 500)
x2_values = calculate_x2(x1_values)
x3_values = calculate_x2_j1(x1_values)
x4_values = calculate_x2_j3(x1_values)


# Calculate the specific points for the callouts
x1_at_x2_100 = (100 - 147.4056772) / -1.265782808  # Solve for x1 when x2 = 100
x2_at_x1_100 = calculate_x2(100)  # Calculate x2 when x1 = 100
x1_eq_x2 = (147.4056772) / (1 + 1.265782808)  # Solve for x1 when x1 = x2

# Calculate the specific points for the callouts
x1_at_x2_100_j1 = (100 - 176.6759548) / -1.265782808  # Solve for x1 when x2 = 100
x2_at_x1_100_j1 = calculate_x2_j1(100)  # Calculate x2 when x1 = 100
x1_eq_x2_j1 = (176.6759548) / (1 + 1.265782808)  # Solve for x1 when x1 = x2

# Calculate the specific points for the callouts
x1_at_x2_100_j2 = (100 - 354.4410007) / -1.265782808  # Solve for x1 when x2 = 100
x2_at_x1_100_j2 = calculate_x2_j3(100)  # Calculate x2 when x1 = 100
x1_eq_x2_j2 = (354.4410007) / (1 + 1.265782808)  # Solve for x1 when x1 = x2




# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x1_values, x3_values, label=r'LCOH=5.290 (j=1)', color='r')
plt.plot(x1_values, x4_values, label=r'LCOH=4.052 (j=3)', color='g')
plt.plot(x1_values, x2_values, label=r'LCOH=5.494 (j=4)', color='b')




# Add scatter points for the specific callouts
plt.scatter([100, x1_at_x2_100, x1_eq_x2], [x2_at_x1_100, 100, x1_eq_x2], color='red')
plt.scatter([100, x1_at_x2_100_j1, x1_eq_x2_j1], [x2_at_x1_100_j1, 100, x1_eq_x2_j1], color='red')
plt.scatter([100, x1_at_x2_100_j2, x1_eq_x2_j2], [x2_at_x1_100_j2, 100, x1_eq_x2_j2], color='red')



# Annotate the points
plt.annotate(f'({100:.0f}, {x2_at_x1_100:.0f})', (100, x2_at_x1_100), textcoords="offset points", xytext=(-20,-15), ha='center')
plt.annotate(f'({x1_at_x2_100:.0f}, 100)', (x1_at_x2_100, 100), textcoords="offset points", xytext=(-20,-15), ha='center')
plt.annotate(f'({x1_eq_x2:.0f}, {x1_eq_x2:.0f})', (x1_eq_x2, x1_eq_x2), textcoords="offset points", xytext=(-20,-15), ha='center')

# Annotate the points
plt.annotate(f'({100:.0f}, {x2_at_x1_100_j1:.0f})', (100, x2_at_x1_100_j1), textcoords="offset points", xytext=(20,15), ha='center')
plt.annotate(f'({x1_at_x2_100_j1:.0f}, 100)', (x1_at_x2_100_j1, 100), textcoords="offset points", xytext=(20,15), ha='center')
plt.annotate(f'({x1_eq_x2_j1:.0f}, {x1_eq_x2_j1:.0f})', (x1_eq_x2_j1, x1_eq_x2_j1), textcoords="offset points", xytext=(20,15), ha='center')

# Annotate the points
plt.annotate(f'({100:.0f}, {x2_at_x1_100_j2:.0f})', (100, x2_at_x1_100_j2), textcoords="offset points", xytext=(20,15), ha='center')
plt.annotate(f'({x1_at_x2_100_j2:.0f}, 100)', (x1_at_x2_100_j2, 100), textcoords="offset points", xytext=(20,15), ha='center')
plt.annotate(f'({x1_eq_x2_j2:.0f}, {x1_eq_x2_j2:.0f})', (x1_eq_x2_j2, x1_eq_x2_j2), textcoords="offset points", xytext=(20,15), ha='center')




plt.title('Discount required to reach competitor')
plt.xlabel('Storage cost reduction [%]')
plt.ylabel('Transport cost reduction [%]')
plt.xlim([0, 350])
plt.ylim([0, 350])
plt.grid(True)
plt.legend()
plt.show()


#%%
# Define the linear equation x2 = - 0.117366647 * x1 + 101.75063696885 (j=3)



# def calculate_x2(x1):
#     return -0.117366647 * x1 + 101.75063696885

# def calculate_x2_j4(x1):
#     return -1.265782808 * x1 + 176.6759548

def calculate_x2_j3(x1):
    return -0.117366647 * x1 + 101.75063696885

# Generate a range of x1 values
x1_values = np.linspace(0, 1000, 500)
# x2_values = calculate_x2(x1_values)
# x3_values = calculate_x2_j1(x1_values)
x4_values = calculate_x2_j3(x1_values)


# # Calculate the specific points for the callouts
# x1_at_x2_100 = (100 - 101.75063696885) / -0.117366647  # Solve for x1 when x2 = 100
# x2_at_x1_100 = calculate_x2(100)  # Calculate x2 when x1 = 100
# x1_eq_x2 = (147.4056772) / (1 + 1.265782808)  # Solve for x1 when x1 = x2

# # Calculate the specific points for the callouts
# x1_at_x2_100_j1 = (100 - 176.6759548) / -1.265782808  # Solve for x1 when x2 = 100
# x2_at_x1_100_j1 = calculate_x2_j1(100)  # Calculate x2 when x1 = 100
# x1_eq_x2_j1 = (176.6759548) / (1 + 1.265782808)  # Solve for x1 when x1 = x2

# Calculate the specific points for the callouts
x1_at_x2_100_j2 = (100 - 101.75063696885) / -0.117366647  # Solve for x1 when x2 = 100
x2_at_x1_100_j2 = calculate_x2_j3(100)  # Calculate x2 when x1 = 100
x1_eq_x2_j2 = (101.75063696885) / (1 + 0.117366647)  # Solve for x1 when x1 = x2




# Create the plot
plt.figure(figsize=(8, 6))
# plt.plot(x1_values, x3_values, label=r'LCOH=5.290 (j=1)', color='r')
plt.plot(x1_values, x4_values, label=r'LCOH=4.052 (j=3)', color='g')
# plt.plot(x1_values, x2_values, label=r'LCOH=5.494 (j=4)', color='b')




# Add scatter points for the specific callouts
# plt.scatter([100, x1_at_x2_100, x1_eq_x2], [x2_at_x1_100, 100, x1_eq_x2], color='red')
# plt.scatter([100, x1_at_x2_100_j1, x1_eq_x2_j1], [x2_at_x1_100_j1, 100, x1_eq_x2_j1], color='red')
plt.scatter([100, x1_at_x2_100_j2, x1_eq_x2_j2], [x2_at_x1_100_j2, 100, x1_eq_x2_j2], color='red')



# # Annotate the points
# plt.annotate(f'({100:.0f}, {x2_at_x1_100:.0f})', (100, x2_at_x1_100), textcoords="offset points", xytext=(-20,-15), ha='center')
# plt.annotate(f'({x1_at_x2_100:.0f}, 100)', (x1_at_x2_100, 100), textcoords="offset points", xytext=(-20,-15), ha='center')
# plt.annotate(f'({x1_eq_x2:.0f}, {x1_eq_x2:.0f})', (x1_eq_x2, x1_eq_x2), textcoords="offset points", xytext=(-20,-15), ha='center')

# # Annotate the points
# plt.annotate(f'({100:.0f}, {x2_at_x1_100_j1:.0f})', (100, x2_at_x1_100_j1), textcoords="offset points", xytext=(20,15), ha='center')
# plt.annotate(f'({x1_at_x2_100_j1:.0f}, 100)', (x1_at_x2_100_j1, 100), textcoords="offset points", xytext=(20,15), ha='center')
# plt.annotate(f'({x1_eq_x2_j1:.0f}, {x1_eq_x2_j1:.0f})', (x1_eq_x2_j1, x1_eq_x2_j1), textcoords="offset points", xytext=(20,15), ha='center')

# Annotate the points
plt.annotate(f'({100:.0f}, {x2_at_x1_100_j2:.0f})', (100, x2_at_x1_100_j2), textcoords="offset points", xytext=(40,-10), ha='center')
plt.annotate(f'({x1_at_x2_100_j2:.0f}, 100)', (x1_at_x2_100_j2, 100), textcoords="offset points", xytext=(20,10), ha='center')
plt.annotate(f'({x1_eq_x2_j2:.0f}, {x1_eq_x2_j2:.0f})', (x1_eq_x2_j2, x1_eq_x2_j2), textcoords="offset points", xytext=(35,5), ha='center')




plt.title('Discount required to reach competitor')
plt.xlabel('Conversion cost reduction [%]')
plt.ylabel('Reconversion cost reduction [%]')
plt.xlim([0, 1000])
plt.ylim([0, 150])
plt.grid(True)
plt.legend()
plt.show()
