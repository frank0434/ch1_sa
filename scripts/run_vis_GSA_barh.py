# %%

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import matplotlib.patches as mpatches
from sklearn import dummy
import config
import pickle
import argparse
import psutil
with open('DummySi_results.pkl', 'rb') as f:
    Dummy_si = pickle.load(f)
# %%
try:
    # Create the parser
    parser = argparse.ArgumentParser(description='Run GSA simulations.')
    # Add the arguments
    parser.add_argument('--GSA_sample_size', type=int, help='The GSA sample size')
    parser.add_argument('--CPUs', type=int, help='The number of CPUs')
    # Parse the arguments
    args = parser.parse_args()
    # Get the GSA_sample_size
    GSA_sample_size = args.GSA_sample_size
    # Get the number of CPUs
    CPUs = args.CPUs
except:
    # Set default values if parsing command line arguments fails
    GSA_sample_size = config.GSA_sample_size
    CPUs = psutil.cpu_count(logical=False)
config.set_variables(GSA_sample_size)

def load_data(day, output_var='TWSO', method='Saltelli'):
    # Load data
    with open(f'{config.p_out_daySi}/{method}_{day}_{output_var}.pkl', 'rb') as f:
        Si = pickle.load(f)
    return Si
# %%
def to_df(self):
    """Convert dict structure into Pandas DataFrame."""
    keys_to_include = ['S1', 'S1_conf', 'ST', 'ST_conf']
    return pd.DataFrame(
        {k: v for k, v in self.items() if k in keys_to_include}, index = config.params_of_interests
    )

# %%
differences = config.days_s2
samplesize = 32768
config.set_variables(samplesize, local=True)
cols = len(differences)
#%%
width = cols * 2.5
scale_factor = 10
saltelli_thres = 0.1 # threshold for saltelli from Wang et al. 2013 or 0.15 from Vanuytrecht et al. 2014
fig, axs = plt.subplots(3, cols, figsize=(width, 9), sharex=True)
colors = [ 'red', 'blue']  # Replace with your actual colors

plt.subplots_adjust(wspace=0.2, hspace=0.05)
for j, var in enumerate(['DVS','LAI','TWSO']):
    for i, d in enumerate(differences):
        # Load data
        Si = load_data(d, var, 'Saltelli')
        df = to_df(Si[f'si_day_{d}_{var}'])


        if i == 0:
        # Sort DataFrame by 'S1' column in descending order
            df_sorted = df.sort_values(by='S1', ascending=True)
            order = df_sorted.index 
            conf_cols = df_sorted.columns.str.contains('_conf')
            confs = df_sorted.loc[:, conf_cols]
            confs.columns = [c.replace('_conf', "") for c in confs.columns]
            Sis = df_sorted.loc[:, ~conf_cols]
            axs[j, i].axvline(x=saltelli_thres, color='r', linestyle='-')

            # Plot sorted DataFrame
            barplot = Sis.plot(kind='barh', yerr=confs * scale_factor, ax=axs[j, i], width = 0.9,
                               legend=False, color = colors)
        if i > 0:
            df_sorted_new = df.reindex(order)
            conf_cols = df_sorted_new.columns.str.contains('_conf')
            confs = df_sorted_new.loc[:, conf_cols]
            confs.columns = [c.replace('_conf', "") for c in confs.columns]
            Sis = df_sorted_new.loc[:, ~conf_cols]
            axs[j, i].axvline(x=saltelli_thres, color='r', linestyle='-')

            # Plot sorted DataFrame
            barplot = Sis.plot(kind='barh', yerr=confs * scale_factor, ax=axs[j, i], 
                               width = 0.9,legend=False, color = colors)

        column_sums = df.loc[:, ~conf_cols].sum().round(2)
        if i > 0:
            axs[j, i].set_yticklabels([])
            axs[j, i].set_yticks([])
        # Get handles and labels of original legend
        
        handles, labels = barplot.get_legend_handles_labels()
  
        # Create custom legend elements and add them to handles and labels
        for col, sum, color in zip(column_sums.keys(), column_sums.values, colors):

            line = plt.Line2D([0], [0], color = color, lw=4, label=f'{col}: {sum:.2f}')
            handles.append(line)
            labels.append(f'{col} Sum: {sum:.2f}')
        # Add combined legend to plot
        # Reverse handles and labels
        axs[j, i].legend(handles=handles[2:][::-1], labels=labels[2:][::-1])
        if i == 0:
            axs[j, i].set_ylabel(var, size=16, weight='bold')
        if j == 0:
            axs[j, i].text(-0, 1.01, str(d) + " DAP", transform=axs[j, i].transAxes, size=16, weight='bold')
plt.xlim(0, 1)
# fig.legend(handles=handles[:2], labels=labels[:2], loc='center right')
fig.text(0.5, .05, 'Sensitivity index', ha='center', size = 16, weight='bold')
scenario = "NL_" if config.run_NL_conditions else ""
filenm = f'{config.p_out}/{scenario}ParameterRank_Saltelli_days_{differences}.svg' 
plt.savefig(filenm, bbox_inches='tight')
plt.show()

# %% # pAWN
method='PAWN'
sortby='median'
dummy = Dummy_si[1][1] # select the upper bound of the dummy si
# Calculate the number of rows for the subplots
cols = len(differences)
width = cols * 2.5
# Create a figure with multiple subplots
fig, axs = plt.subplots(3, cols, figsize=(width, 9), sharex=True)
plt.subplots_adjust(wspace=0.15)
# Loop over the output variables
for j, var in enumerate(['DVS','LAI','TWSO']):
    # Loop over the days and axes
    for i, d in enumerate(differences):
        # Load data
        data = load_data(d, var, 'PAWN')
        df = data[f'si_day_{d}_{var}'].to_df()
        # Sort DataFrame by 'median' column in descending order
        if i == 0:
            df_sorted = df.sort_values(by=sortby, ascending=True)
            order = df_sorted.index
            column_sums = df.sum().round(2).drop(['minimum', "maximum",'CV'])
            # Plot sorted DataFrame
            barplot = df_sorted[sortby].plot(kind='barh', ax=axs[j, i], width=0.5)
            axs[j, i].axvline(x=dummy, color='r', linestyle='-')
        if i > 0:
            df_sorted_new = df.reindex(order)
            column_sums = df_sorted_new.sum().round(2).drop(['minimum', "maximum",'CV'])
            # Plot sorted DataFrame
            barplot = df_sorted_new[sortby].plot(kind='barh', ax=axs[j, i], width=0.5)
            axs[j, i].axvline(x=dummy, color='r', linestyle='-')
        handles, labels = barplot.get_legend_handles_labels()
        
        for col, sum in column_sums.items():
            line = plt.Line2D([0], [0], color='b', lw=4, label=f'{col}: {sum:.2f}')
            handles.append(line)
            labels.append(f'{col} Sum: {sum:.2f}')
        # Add combined legend to plot
        axs[j, i].legend(handles=handles[1:], labels=labels[1:])
        # Remove y-axis for 2nd, 3rd, 4th, etc. column subplots
        if i > 0:
            axs[j, i].set_yticklabels([])
            axs[j, i].set_yticks([])
        # Add label to top row only
        if j == 0:
            axs[j, i].text(-0, 1.01, chr(65 + i )+ ') ' +str(d) + " DAP", transform=axs[j, i].transAxes, size=20, weight='bold')
        # Add y-label for each row
        if i == 0:
            axs[j, i].set_ylabel(var, fontsize=20)
# Add x-label to the entire figure
# Add y-label to the entire figure
plt.tight_layout()
plt.xlim(0, 1)
fig.text(0.5, -.01, 'Sensitivity index', ha='center', size = 16, weight='bold')
scenario = "NL_" if config.run_NL_conditions else ""
filenm = f'{config.p_out}/{scenario}ParameterRank_PAWN_days_{differences}.svg' 
plt.savefig(filenm, bbox_inches='tight')
plt.show()
# %%

width = cols * 2.5
fig, axs = plt.subplots(3, cols, figsize=(width, 10), sharex=True)
plt.subplots_adjust(wspace=0.4, hspace=0.1)
for j, var in enumerate(['DVS','LAI','TWSO']):
    for i, d in enumerate(differences):
        # Load data
        Si = load_data(d, var, 'Saltelli')
        df = to_df(Si[f'si_day_{d}_{var}'])

        # Sort DataFrame by 'S1' column in descending order
        df_sorted = df.sort_values(by='S1', ascending=True)
        conf_cols = df_sorted.columns.str.contains('_conf')
        confs = df_sorted.loc[:, conf_cols]
        confs.columns = [c.replace('_conf', "") for c in confs.columns]
        Sis = df_sorted.loc[:, ~conf_cols]

        # Plot sorted DataFrame
        barplot = Sis.plot(kind='barh', yerr=confs, ax=axs[j, i])
        # if i > 0:
        #     axs[j, i].set_yticklabels([])
        #     axs[j, i].set_yticks([])
        # Create custom legend
        column_sums = df.loc[:, ~conf_cols].sum().round(2)

        # Get handles and labels of original legend
        handles, labels = barplot.get_legend_handles_labels()

        # Create custom legend elements and add them to handles and labels
        for col, sum in column_sums.items():
            line = plt.Line2D([0], [0], color='b', lw=4, label=f'{col}: {sum:.2f}')
            handles.append(line)
            labels.append(f'{col} Sum: {sum:.2f}')

        # Add combined legend to plot
        axs[j, i].legend(handles=handles, labels=labels, loc='lower right')
        if i == 0:
            axs[j, i].set_ylabel(var, size=16, weight='bold')
        if j == 0:
            axs[j, i].text(-0, 1.01, str(d) + " DAP", transform=axs[j, i].transAxes, size=16, weight='bold')
plt.xlim(0, 1)
plt.savefig(f'{config.p_out}/ParameterRank_Saltelli_days_{differences}.svg', bbox_inches='tight')
plt.show()

# %%
def plot_sorted_df(day, output_var=['TWSO'], method='Saltelli', samplesize=32768, sortby='S1'):
    # Check if day is a list
    if isinstance(day, list) and isinstance(output_var, list):
        # Calculate the number of rows for the subplots
        num_rows = len(day)

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(len(output_var), len(day), figsize=(14, 14), sharex=True)
        plt.subplots_adjust(wspace=0.15)

        # Loop over the output variables
        for j, var in enumerate(output_var):
            # Loop over the days and axes
            for i, d in enumerate(day):
                # Load data
                print(f'day {d}; output of interest:{var}')
                with open(f'{config.p_out_daySi}/{method}_{d}_{output_var}.pkl', 'rb') as f:
                    Si = pickle.load(f)

                df = to_df(Si[f'si_day_{d}_{var}'])

                # Sort DataFrame by 'S1' column in descending order
                df_sorted = df.sort_values(by=sortby, ascending=True)
                conf_cols = df_sorted.columns.str.contains('_conf')
                confs = df_sorted.loc[:, conf_cols]
                confs.columns = [c.replace('_conf', "") for c in confs.columns]
                Sis = df_sorted.loc[:, ~conf_cols]

                # Plot sorted DataFrame
                barplot = Sis.plot(kind='barh', yerr=confs, ax=axs[j, i])

                # Create custom legend
                column_sums = df.loc[:, ~conf_cols].sum().round(2)

                # Get handles and labels of original legend
                handles, labels = barplot.get_legend_handles_labels()

                # Create custom legend elements and add them to handles and labels
                for col, sum in column_sums.items():
                    line = plt.Line2D([0], [0], color='b', lw=4, label=f'{col}: {sum:.2f}')
                    handles.append(line)
                    labels.append(f'{col} Sum: {sum:.2f}')

                # Add combined legend to plot
                axs[j, i].legend(handles=handles, labels=labels)
                axs[j, i].text(-0, 1.01, chr(65 + i )+ ') ' +str(d) + " DAP", transform=axs[j, i].transAxes, size=16, weight='bold')
        plt.xlim(0, 1)
        # Add y-label to the entire figure
        # Save the figure
        plt.savefig(f'{config.p_out}/ParameterRank_{method}_{output_var}_days_{day}.svg', dpi=300, bbox_inches='tight')
        plt.clf()  # Clear figure to avoid overlapping plots

    else:
        # The rest of the code remains the same as it handles the case when day is not a list
        # Load data
        Si = load_data(day, output_var, method)
        df = to_df(Si[f'si_day_{day}_{output_var}'])

        # Sort DataFrame by 'S1' column in descending order
        df_sorted = df.sort_values(by=sortby, ascending=True)
        conf_cols = df_sorted.columns.str.contains('_conf')
        confs = df_sorted.loc[:, conf_cols]
        confs.columns = [c.replace('_conf', "") for c in confs.columns]
        Sis = df_sorted.loc[:, ~conf_cols]

        # Plot sorted DataFrame
        barplot = Sis.plot(kind='barh', yerr=confs)
        plt.title(f'Day {day} {output_var} {method} method with samplesize  {samplesize}. Sorted by S1')
        plt.xlabel('Parameter name')
        plt.ylabel('Sensitivity index')

        # Create custom legend
        column_sums = df.loc[:, ~conf_cols].sum().round(2)

        # Get handles and labels of original legend
        handles, labels = barplot.get_legend_handles_labels()

        # Create custom legend elements and add them to handles and labels
        for col, sum in column_sums.items():
            line = plt.Line2D([0], [0], color='b', lw=4, label=f'{col}: {sum:.2f}')
            handles.append(line)
            labels.append(f'{col} Sum: {sum:.2f}')
        plt.xlim(0, 1)
        # Add combined legend to plot
        plt.legend(handles=handles, labels=labels)

        # Save the figure
        plt.savefig(f'{config.p_out}/ParameterRank_{method}_{output_var}_day_{day}.png', dpi=300, bbox_inches='tight')
        plt.clf()  # Clear figure to avoid overlapping plots
# %%

plot_sorted_df(day = differences, output_var='DVS', method='Saltelli')
# %%

def plot_sorted_df_pawn(day, output_var='TWSO', method='PAWN', sortby='median', dummy = Dummy_si):
    # Check if day is a list
    if isinstance(day, list):
        # Calculate the number of rows for the subplots
        num_rows = len(day)

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
        # plt.suptitle(f'Day {day} {output_var} {method} method with samplesize  {samplesize}. Sorted by {sortby}')
        # plt.subplots_adjust(top=0.8)  # Adjust this value to your liking
      

        # Loop over the days and axes
        for i, ( d, ax )in enumerate(zip(day, axs.flatten())):
            # Load data
            data = load_data(d, output_var, method)   

            df = data[f'si_day_{d}_{output_var}'].to_df()
            # df.loc['Dummy', ['median']] = [round(dummy[0], 3)]
            # df['median'] = df['median'].astype(float)

            # Sort DataFrame by 'median' column in descending order
            df_sorted = df.sort_values(by=sortby, ascending=True)
            column_sums = df.sum().round(2).drop(['minimum', "maximum",'CV'])

            # Plot sorted DataFrame
            barplot = df_sorted[sortby].plot(kind='barh', ax=ax,width=0.5)
            ax.axvline(x=dummy[1][1], color='r', linestyle='-')
            handles, labels = barplot.get_legend_handles_labels()
            for col, sum in column_sums.items():
                line = plt.Line2D([0], [0], color='b', lw=4, label=f'{col}: {sum:.2f}')
                handles.append(line)
                labels.append(f'{col} Sum: {sum:.2f}')
            # Add combined legend to plot
            ax.legend(handles=handles, labels=labels, loc='lower right')
            ax.text(-0, 1.01, chr(65 + i )+ ') ' +str(d) + " DAP", transform=ax.transAxes, size=16, weight='bold')
        # Add x-label to the entire figure
        fig.text(0.5, 0, 'Sensitivity index', ha='center')

        # Add y-label to the entire figure
        fig.text(-0.01, 0.5, 'Parameter name', va='center', rotation='vertical')
        plt.tight_layout()
        plt.xlim(0, 1)
        # plt.show()
        plt.savefig(f'{config.p_out}/ParameterRank_{method}_{output_var}_days_{day}.svg', dpi=500, bbox_inches='tight')
        plt.clf() # Clear figure to avoid overlapping plots

    else:
        # Load data
        data = load_data(day, output_var, method)   

        df = data[f'si_day_{day}_{output_var}'].to_df()

        # Sort DataFrame by 'median' column in descending order
        df_sorted = df.sort_values(by=sortby, ascending=True)
        column_sums = df.sum().round(2)

        # Plot sorted DataFrame
        barplot = df_sorted[sortby].plot(kind='barh')
        handles, labels = barplot.get_legend_handles_labels()
        for col, sum in column_sums.items():
            line = plt.Line2D([0], [0], color='b', lw=4, label=f'{col}: {sum:.2f}')
            handles.append(line)
            labels.append(f'{col} Sum: {sum:.2f}')
        # Add combined legend to plot
        plt.legend(handles=handles, labels=labels)
        plt.title(f'Day {day} {output_var} {method} method with samplesize  {samplesize}. Sorted by {sortby}')
        plt.xlabel('Parameter name')
        plt.ylabel('Sensitivity index')
        plt.xlim(0, 1)
        # plt.show()
        plt.savefig(f'{config.p_out}/ParameterRank_{method}_{output_var}_day_{day}.svg', dpi=300, bbox_inches='tight')
        
        plt.clf() # Clear figure to avoid overlapping plots

# %%

# %%
def plot_scatter_with_legend(day, output_var, method, pair = ['S1', 'median']):
    # Load data
    Si_Saltelli = load_data(day, output_var, 'Saltelli')
    Si_PAWN = load_data(day, output_var, method)

    df_Saltelli = to_df(Si_Saltelli[f'si_day_{day}_{output_var}'])
    df_PAWN = Si_PAWN[f'si_day_{day}_{output_var}'].to_df()

    # Merge dataframes
    merged_df = df_Saltelli.merge(df_PAWN, left_index=True, right_index=True)

    # Map the index to the color map
    colors = merged_df.index.map(lambda x: config.name_color_map[x]).to_list()

    # Plot with colors based on index
    scatter = plt.scatter(merged_df[pair[0]], merged_df[pair[1]], c=colors)

    # Create a list of unique index values
    unique_indices = merged_df.index.unique()

    # Create a list of patches for the legend
    patches = [mpatches.Patch(color=config.name_color_map[i], label=i) for i in unique_indices]

    # Add legend manually
    plt.legend(handles=patches, loc= (1.01, 0), title="Parameter name")
    plt.xlabel('Main effect sensitivity index - Saltelli')
    plt.ylabel('Median sensitivity index - PAWN')
    plt.title(f'Day {day} {output_var} sensitivity index comparison')

    # Set x and y axes to start from 0 and end at the maximum value of x or y
    max_value = max(merged_df['S1'].max(), merged_df['median'].max()) + 0.1
    plt.xlim(0, max_value )
    plt.ylim(0, max_value )
    # Add a 1:1 line
    plt.plot([0, max_value], [0, max_value], 'k--')
    plt.savefig(f'{config.p_out}/ParameterCompare_{output_var}_day_{day}.png', 
                dpi=300, bbox_inches='tight')
    plt.clf() # Clear figure to avoid overlapping plots

# %%

def plot_samplesize_effect(start=5, end=16, index='te', day = 38, output_var='TWSO'):
    # Initialize an empty DataFrame to store the results
    combined = pd.DataFrame()
    combined_pawn = pd.DataFrame()
    totalNo_ofsims = [2 ** x * (len(config.params_of_interests) * 2+ 2) for x in range(start, end)]

    # Loop over the range of powers of 2 
    for ss in [2 ** x for x in range(start, end)]:
        # Load the data - load saltelli data
        with open(f'{config.p_out}/daySi_{ss}/Saltelli_{day}_{output_var}.pkl', 'rb') as f:
            Si = pickle.load(f)

        # Convert to DataFrame and reset index
        saltelli_df = to_df(Si[f'si_day_{day}_{output_var}']).reset_index()
        saltelli_df.loc[:, 'SampleSize'] = ss

        # Filter rows where 'index' column is 'te'
        te = saltelli_df.loc[saltelli_df['index'] == index, :]

        # Append to the combined DataFrame
        combined = pd.concat([combined, te])
        # Load the data - load pawn data
        with open(f'{config.p_out}/daySi_{ss}/PAWN_{day}_{output_var}.pkl', 'rb') as f:
            Pawn = pickle.load(f)

        # Convert to DataFrame and reset index
        pawn_df = Pawn[f'si_day_{day}_{output_var}'].to_df().reset_index()
        pawn_df.loc[:, 'SampleSize'] = ss

        # Filter rows where 'index' column is 'te'
        te_pawn = pawn_df.loc[pawn_df['index'] == index, :]

        # Append to the combined DataFrame
        combined_pawn = pd.concat([combined_pawn, te_pawn])

    # Set 'SampleSize' as the index
    combined.set_index('SampleSize', inplace=True)
    # Set 'SampleSize' as the index
    combined_pawn.set_index('SampleSize', inplace=True)
    combined_pawn.drop(columns=['index','CV'], inplace=True)
    # Create a figure with two subplots (one row, two columns)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)

    # Plot 'S1' on the first subplot
    # axs[0].errorbar(combined.index, combined['S1'], yerr=combined['S1_conf'], fmt='o-', capsize=5)
    # # axs[0].set_title('Si values with confidence intervals')
    # # axs[0].set_xlabel('SampleSize')
    # axs[0].set_ylabel('Main effect (S1)')
    # axs[0].set_xscale('log')
    # axs[0].set_xticks(combined.index)
    # axs[0].set_xticklabels(totalNo_ofsims)

    # Plot 'ST' on the second subplot
    axs[0].errorbar(combined.index, combined['ST'], yerr=combined['ST_conf'], fmt='o-', capsize=5)
    axs[0].set_ylabel('Total effect (ST)')
    axs[0].set_xscale('log')
    axs[0].set_xticks(combined.index)
    axs[0].set_xticklabels(totalNo_ofsims)
    # Plot pawn on the 3rd subplot
    axs[1].plot(combined_pawn.index, combined_pawn['mean'], color='black', linestyle='dashed', label = "mean")    
    
    axs[1].plot(combined_pawn.index, combined_pawn['median'], label = "median")
    axs[1].set_xlabel('Simulations runs')
    axs[1].set_ylabel('PAWN median and mean')
    axs[1].set_xscale('log')
    axs[1].set_xticks(combined_pawn.index)
    axs[1].set_xticklabels(totalNo_ofsims)
    axs[1].legend()
    plt.ylim(0, 1.4)
    plt.savefig(f'{config.p_out}/SampleSizeEffect_{output_var}_day_{day}_{index}.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.p_out}/SampleSizeEffect_{output_var}_day_{day}_{index}.svg',bbox_inches='tight')    

    plt.show()
    plt.close()

#%% 
plot_samplesize_effect(day=105)
# %%

def plot_pawn_effect(start=5, end=16, index='te', day=38, output_var='TWSO'):
    # Initialize an empty DataFrame to store the results
    combined = pd.DataFrame()

    # Loop over the range of powers of 2
    for ss in [2 ** x for x in range(start, end)]:
        # Load the data
        with open(f'{config.p_out}/daySi_{ss}/PAWN_{day}_{output_var}.pkl', 'rb') as f:
            Pawn = pickle.load(f)

        # Convert to DataFrame and reset index
        pawn_df = Pawn[f'si_day_{day}_{output_var}'].to_df().reset_index()
        pawn_df.loc[:, 'SampleSize'] = ss

        # Filter rows where 'index' column is 'te'
        te = pawn_df.loc[pawn_df['index'] == index, :]

        # Append to the combined DataFrame
        combined = pd.concat([combined, te])

    # Set 'SampleSize' as the index
    combined.set_index('SampleSize', inplace=True)
    combined.drop(columns=['index','CV'], inplace=True)

    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop over the columns
    for col in combined.columns:
        ax.plot(combined.index, combined[col], marker='o', label=col)

    # Add a legend
    ax.legend()
    ax.set_title(f'PAWN values with cv for {output_var} on day {day} with {index} as the input')
    ax.set_xlabel('SampleSize')
    ax.set_ylabel('PAWN')
    ax.set_xscale('log')
    ax.set_xticks(combined.index)
    ax.set_xticklabels([2 ** x for x in range(start, end)])

    plt.savefig(f'{config.p_out}/SampleSizeEffect/Pawn_{output_var}_day_{day}_{index}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() # Clear figure to avoid overlapping plots

    # plt.show()
# %%
# config.cols_of_interests
cols_of_interests = ['DVS', 'LAI', 'TWSO']
# %%
GSA_sample_size = 65536
config.set_variables(GSA_sample_size, local=True)
# samplesize = [2 ** x for x in range(5, 16)]
# %%
plot_samplesize_effect()
plot_pawn_effect()
# %%
# 20240213 output for writing
plot_samplesize_effect(start=5, end=17, index='te', day = 105, output_var='TWSO')

# %%

load_data(105, 'TWSO', 'PAWN')['si_day_105_TWSO'].to_df()
# %%
for day in differences:
    for var in ['DVS','LAI','TWSO']:
        for para in config.params_of_interests:
            print(f'day {day}; output of interest:{var}; parameter of interest: {para}')
            plot_pawn_effect(start=5, end=16, index=para, day = day, output_var=var)

# %%

for day in [105]:
    for var in ['DVS','LAI','TWSO']:
        for para in config.params_of_interests:
            print(f'day {day}; output of interest:{var}; parameter of interest: {para}')
            plot_samplesize_effect(start=5, end=17, index=para, day = day, output_var=var)


# %%  # all the ranking graphs 
differences = calculate_days_difference(planting, harvest)
samplesize = GSA_sample_size
plot_sorted_df(differences, 'DVS', 'Saltelli')
# %%
# produce all the graphs
for day in differences:
    for var in config.cols_of_interests:
        print(f'day {day}; output of interest:{var}')
        plot_sorted_df(day, var, 'Saltelli')

# %%
# Call the function for day 53 and 105
for day in differences:
    for var in ['DVS','LAI','TWSO']:
        print(f'day {day}; output of interest:{var}')
        plot_sorted_df_pawn(day, var, 'PAWN')
# %%
for day in differences:
    for var in ['DVS','LAI','TWSO']:
        print(f'day {day}; output of interest:{var}')
        plot_scatter_with_legend(day, var, 'PAWN')
# %%

# %% 
samplesize = 32768
for var in ['DVS', 'LAI', 'TWSO']:
    print(f'day {differences}; output of interest:{var}')
    plot_sorted_df_pawn(differences, var, 'PAWN')
# %%
samplesize = 32768
for var in ['DVS', 'LAI', 'TWSO']:
    print(f'day {differences}; output of interest:{var}')
    plot_sorted_df(differences, var, 'Saltelli')



#%%
d = 105
output_var = 'TWSO'
method = 'PAWN'
sortby = 'median'
dummy = Dummy_si
data = load_data(d, output_var, method)   
df = data[f'si_day_{d}_{output_var}'].to_df()
df
# %%
df.loc['Dummy', ['median']] = [round(dummy[0], 3)]
df['median'] = df['median'].astype(float)
df
# %%
# Sort DataFrame by 'median' column in descending order
df_sorted = df.sort_values(by=sortby, ascending=True)
column_sums = df.sum().round(2).drop(['minimum', "maximum",'CV'])