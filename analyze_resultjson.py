#%%
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Step 1 & 2: Parse the hyperparameters from the filenames and store them in a dictionary
# folder_path = 'experiments/ucf101-shot_16-seed_1/RN50/regularization_text_0_hand_crafted-image_0_flip_view_1/linear_zeroshot/logit_4.60517/'    #regularization_ with beta
folder_path = 'experiments/ucf101-shot_16-seed_1/RN50/text_0_hand_crafted-image_0_flip_view_1/linear_zeroshot/logit_4.60517/'   #normal cross-modal training
json_files_path = [os.path.join(folder_path, dir, dir + '.json') for dir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, dir))]
json_files_names = [os.path.basename(file) for file in json_files_path]

data = []
for i, file in enumerate(json_files_names):
    params = file.split('-')[1:-1]  # remove the 'optim_adamw' and the trailing '-'
    param_dict = {param.split('_')[0]: float(param.split('_')[1]) for param in params}
    
    # Step 3: Load the JSON file and add the data to the dictionary
    with open(json_files_path[i], 'r') as f:
        json_dict = json.load(f)
        for key, value in json_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    param_dict[k] = v
            else:
                param_dict[key] = value
    
    data.append(param_dict)

# Step 4: Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)
#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#----------------------settings----------------------
# Select the rows where beta == 1.5 and bs == 32
selected_rows = df[(df['bs'] == 32) & (df['lr'] == 0.0001)]   #(df['beta'] == 1.5) & (df['bs'] == 32) & (df['lr'] == 0.0001) & (df['wd'] == 0.01) 
# Step 5: Group the DataFrame by the hyperparameter and calculate the mean of the performance metric
grouped_var = 'wd'
# compar_var = 'head_wiseft_0.5'
compar_var = 'head'
#----------------------settings----------------------

grouped_name = selected_rows.groupby(grouped_var)[compar_var].mean()
# grouped_name = df.groupby(grouped_var)[compar_var].mean()

# Step 6: Visualize the results
ax = grouped_name.plot(kind='bar', color='lightblue')
# Highlight the bar with the maximum value
ax.patches[grouped_name.values.argmax()].set_facecolor('r')

# Set the x and y labels
plt.xlabel(f'{grouped_var}')
plt.ylabel(f'{compar_var}')

# Set the title
plt.title(f'{compar_var} VS {grouped_var}')

# Add grid
plt.grid(True, which='both', color='gray', linewidth=0.5)

# Customize the y ticks
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(10))  # Set the number of major ticks
plt.gca().yaxis.set_minor_locator(ticker.MaxNLocator(50))  # Set the number of minor ticks

# Set the y limit
plt.ylim(grouped_name.min() - 0.1, grouped_name.max() + 0.1)
# Show the y values on top of the bars
for i, v in enumerate(grouped_name.values):
    ax.text(i, v + 0.01, "{:.3f}".format(v), ha='center')
plt.show()


# %%
#conclusion for rugulizaion: 
# 1. beta= 1.5 > 1.0 > 0.5 > 0.0
# 2. head > head_wiseft_0.5 
# 3. wd: 0.01 > 0.0001 > 0 (for head_wiseft_0.5 it is reversed)
# 4. lr: head: 0.0001 > 0.001 (for head_wiseft_0.5 it is reversed)
# 5. bs:  32 > 8 
# 6. iter: ACC increase, then decrease with iter increasing, best is around 1400


#best params for head is: beta=1.5(larger better), wd=0.01(not so important), lr=0.0001, bs=32(larger better)