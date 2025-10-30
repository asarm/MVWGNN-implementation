import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load the data
df = pd.read_csv('data/wind_speed.csv')["Vancouver"]


debug_target = torch.load('debug_target.pt', map_location=torch.device('cpu'))
debug_hist = torch.load('debug_hist.pt', map_location=torch.device('cpu'))
target_npy = debug_target.numpy()
hist_npy = debug_hist.numpy()
print(f"Loaded target shape: {target_npy.shape}")
print(f"Loaded history shape: {hist_npy[:,:,0].shape}")

station_idx = 1
plt.figure(figsize=(12, 6))

# Tarihsel değerleri çizgi olarak çiz
hist_length = hist_npy.shape[1]
plt.plot(range(hist_length), hist_npy[station_idx, :, 0], label='Historical Wind Speed', linestyle='-', color='blue')

# Target değerini nokta olarak ekle (tek bir değer)
target_x = hist_length  # Tarihsel verinin hemen sonrası
plt.scatter(target_x, target_npy[station_idx], label='Target Wind Speed', marker='o', color='red', s=100)

plt.title(f'Historical vs Target Wind Speed (Station {station_idx})')
plt.xlabel('Time Step')
plt.ylabel('Wind Speed')
plt.legend()
plt.grid()
plt.savefig('data_check_plot.png')