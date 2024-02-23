# %%
import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the CSV file
df = pd.read_csv('../report20240125.csv', delim_whitespace=True)

df
# %%
# 2. Filter the completed jobs
df_completed = df[df['State'] == 'COMPLETED']
df_completed.to_csv('../report20240125_completed.csv', index=False)
# 3. Select JobName, JobID, and CPUTimeRAW
df_selected = df_completed[['JobName', 'JobID', 'AllocCPUS','CPUTimeRAW']]

# Convert CPUTimeRAW from seconds to hours
df_selected['CPUTimeHour'] = df_selected['CPUTimeRAW'] / 3600
df_selected['TimePerCPU'] = df_selected['CPUTimeHour'] / df_selected['AllocCPUS']

# 4. Plot the CPU time based on JobID
plt.figure(figsize=(10,6))
plt.scatter(df_selected['CPUTimeHour'], df_selected['TimePerCPU'], c=df_selected['AllocCPUS'], cmap='viridis', marker='o')
plt.colorbar(label='AllocCPUS')
plt.title('CPU Time by Job ID')
plt.grid(True)
plt.show()
# %%
