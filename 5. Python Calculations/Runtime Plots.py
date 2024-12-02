# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:50:03 2024

@author: spide
"""

import matplotlib.pyplot as plt


def filter_runtimes(Runtime_List, threshold=10):
    return [runtime for runtime in Runtime_List if runtime <= threshold]

Runtime_List_Reduced = filter_runtimes(Runtime_List)

def upper_runtimes(Runtime_List, threshold=10):
    return [runtime for runtime in Runtime_List if runtime >= threshold]

Runtime_List_Upper = upper_runtimes(Runtime_List)


def plot_runtime_distribution(Runtime_List):
    # Create a histogram for the runtime data
    plt.hist(Runtime_List, bins=20, edgecolor='black', color='skyblue')

    # Adding labels and title
    plt.title('Frequency Distribution of Runtimes')
    plt.xlabel('Runtimes')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

# Example usage
# Runtime_List = [12.5, 14.3, 12.7, 13.8, 15.0, 16.1, 15.6, 12.1, 14.4, 15.2, 12.9, 16.8]
plot_runtime_distribution(Runtime_List_Reduced)
plot_runtime_distribution(Runtime_List_Upper)
#%% Boxplot

def plot_boxplot(Runtime_List):
    # Create a box-and-whiskers plot for the runtime data
    plt.boxplot(Runtime_List, patch_artist=True, boxprops=dict(facecolor='skyblue'))

    # Adding labels and title
    plt.title('Box and Whiskers Plot of Runtimes')
    plt.ylabel('Runtimes')

    # Show the plot
    plt.show()
    
plot_boxplot(Runtime_List)
plot_boxplot(Runtime_List_Reduced)

#%% Outliers


def count_outliers(data):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Calculate bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return len(outliers), outliers

# Example usage
# Runtime_List = [12.5, 14.3, 12.7, 13.8, 15.0, 16.1, 15.6, 12.1, 14.4, 15.2, 12.9, 16.8, 23.0, 25.4]
outlier_count, outlier_values = count_outliers(Runtime_List_Reduced)

print("Number of outliers:", outlier_count)
print("Outliers:", outlier_values)
min(outlier_values)
