import math

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

# %%
import matplotlib.pyplot as plt

# Define cold and warm colors
cold_colors = ['#add8e6', '#87ceeb', '#00ffff', '#008080', '#90ee90', '#e6e6fa', '#0000ff', '#00008b']
warm_colors = ['#ffffe0', '#ffcccb', '#ff7f50', '#ffa500', '#ff6347', '#ff0000', '#ff8c00']

# Create a combined list for plotting
all_colors = cold_colors + warm_colors
labels = [f'Cold {i+1}' for i in range(len(cold_colors))] + [f'Warm {i+1}' for i in range(len(warm_colors))]

# Plotting the colors
plt.figure(figsize=(10, 5))
plt.barh(labels, [1] * len(all_colors), color=all_colors)
plt.xlim(0, 1)  # Adjust x limit to fit color bars
plt.title('Cold and Warm Colors Visualization')
plt.axis('off')  # Turn off the axis
plt.show()
# %%
from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 100, 15)[None, :]
plt.imshow(x, aspect='auto', cmap=cm.batlow) # or any other colourmap
plt.axis('off')
plt.show()

cm.batlow.colors  # this is a list of the RGB values of the colours in the colormap
# %%
from cmcrameri import show_cmaps

show_cmaps()
# %%
from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb2hex

# Get the RGB values from the batlow colormap
batlow_rgb = cm.berlin.colors

# Convert RGB values to hex codes
batlow_hex = [rgb2hex(rgb) for rgb in batlow_rgb]

# Print the hex codes
print("Batlow colormap hex codes:")
for i, hex_code in enumerate(batlow_hex):
    print(f"{i}: {hex_code}")

# Create a visualization of the extracted colors
plt.figure(figsize=(12, 3))
for i, color in enumerate(batlow_hex):
    plt.axvspan(i, i+1, color=color)
plt.xlim(0, len(batlow_hex))
plt.axis('off')
plt.title('Extracted batlow color palette')
plt.tight_layout()
plt.show()

# If you need a specific number of colors
num_colors = 8  # Change this to get fewer/more colors
indices = np.linspace(0, len(batlow_hex)-1, num_colors).astype(int)
selected_colors = [batlow_hex[i] for i in indices]

print("\nSelected subset of colors:")
# for i, color in enumerate(selected_colors):
#     print(f"{i}: {color}")

# Visualize the selected colors
plt.figure(figsize=(12, 1))
for i, color in enumerate(selected_colors):
    plt.axvspan(i, i+1, color=color)
plt.xlim(0, len(selected_colors))
plt.axis('off')
plt.title('Selected batlow colors')
plt.tight_layout()
plt.show()
# %%

# List of cool colors
cool_colors = ['#011959',
 '#124761',
 '#2b655e',
 '#627941',
 '#a38a2c',
 '#e69858',
 '#fdb0a9',
 '#faccfa']

# List of warm colors
warm_colors = ['#b4339a',
 '#be558e',
 '#c97583',
 '#d1947a',
 '#d9b572',
 '#e2d66a',
 '#ffff66']

# Example usage: print the color lists
# visualise the colours 
# Plotting the colors
plt.figure(figsize=(10, 5))
plt.barh(range(len(cool_colors)), [1] * len(cool_colors), color=cool_colors)
plt.barh(range(len(cool_colors), len(cool_colors) + len(warm_colors)), [1] * len(warm_colors), color=warm_colors)
plt.xlim(0, 1)  # Adjust x limit to fit color bars
plt.title('Cool and Warm Colors Visualization')
plt.yticks(range(len(cool_colors) + len(warm_colors)), ['Cool ' + str(i+1) for i in range(len(cool_colors))] + ['Warm ' + str(i+1) for i in range(len(warm_colors))])
plt.axis('off')  # Turn off the axis
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# HEX codes
cool_hex = ['#011959',
 '#124761',
 '#2b655e',
 '#627941',
 '#a38a2c',
 '#e69858',
 '#fdb0a9',
 '#faccfa']

warm_hex = ['#b4339a',
 '#be558e',
 '#c97583',
 '#d1947a',
 '#d9b572',
 '#e2d66a',
 '#ffff66']

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Cool colors
for i, color in enumerate(cool_hex):
    ax1.add_patch(patches.Rectangle((i, 0), 1, 1, color=color))
    ax1.text(i + 0.5, 0.5, color, ha='center', va='center', 
             color='white' if i < 4 else 'black', fontsize=8)
ax1.set_xlim(0, len(cool_hex))
ax1.set_title('Cool Colors (cmcrameri.oslo)', fontsize=10)
ax1.axis('off')

# Warm colors
for i, color in enumerate(warm_hex):
    ax2.add_patch(patches.Rectangle((i, 0), 1, 1, color=color))
    ax2.text(i + 0.5, 0.5, color, ha='center', va='center', 
             color='white' if i < 3 else 'black', fontsize=8)
ax2.set_xlim(0, len(warm_hex))
ax2.set_title('Warm Colors (cmcrameri.roma)', fontsize=10)
ax2.axis('off')

plt.tight_layout()
plt.show()
# %%
# List of cool colors
cool_colors = ['#011959',
 '#124761',
 '#2b655e',
 '#627941',
 '#a38a2c',
 '#e69858',
 '#fdb0a9',
 '#faccfa']

# List of warm colors
warm_colors = ['#b4339a',
 '#be558e',
 '#c97583',
 '#d1947a',
 '#d9b572',
 '#e2d66a',
 '#ffff66']

# %%
from cmcrameri import cm
from matplotlib.colors import rgb2hex
import numpy as np
# Get batlow colormap and convert to hex codes
batlow_rgb = cm.batlow.colors
batlow_hex = [rgb2hex(rgb) for rgb in batlow_rgb]

# Select specific number of colors (evenly distributed across the colormap)
num_warm_colors = 8
num_cool_colors = 7
total_colors = num_warm_colors + num_cool_colors

# Extract selected colors
indices = np.linspace(0, len(batlow_hex)-1, total_colors).astype(int)
selected_colors = [batlow_hex[i] for i in indices]

# Split into warm and cool colors (batlow's colormap goes from deep blue to red)
warm_colors = selected_colors[:num_warm_colors]
cool_colors = selected_colors[num_warm_colors:]

# Parameters to be colored (in the order they should receive colors)
main_params = [
    'TSUM1', 'TSUM2', 'SPAN', 'TBASEM', 'TSUMEM', 'TEFFMX', 't1_pheno', 'te_pheno'  # First 8 params
]

secondary_params = [
    'TDWI', 'RGRLAI', 'tm1', 't1', 't2', 'te', 'Q10'  # Last 7 params
]

# Create the name_color_map with automated color assignments
name_color_map = {
    'mean': 'blue',
    'median': 'red',
    'minimum': 'grey',
    'maximum': 'grey',
    'CV': 'green',  # Keep these standard colors
}

# Assign warm colors to the first 8 parameters
for param, color in zip(main_params,cool_colors ):
    name_color_map[param] = color

# Assign cool colors to the next 7 parameters
for param, color in zip(secondary_params, warm_colors):
    name_color_map[param] = color
