import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

game_name = "python_block_dominoes"
results_dir = "open_spiel/python/examples/tiny_block_dominoes/results/h2h/"
agent_choices = ["cfr", "cfrplus", "mmd", "random", "qlearner"]



# Initialize an empty DataFrame
df = pd.DataFrame()

# Load all h2h results between CFR and the other algorithms
for agent in agent_choices:
    if agent != "cfr":
        colname = f"cfr Return Against {agent}"
        temp_df = pd.read_csv(results_dir + f"{game_name}_cfr_{agent}.csv", index_col=0)
        temp_df.columns = [colname, "Order"]
        temp_df = temp_df.drop(columns=["Order"])  # Drop the "Order" column
        temp_df["Agent"] = agent  # Add a new column for the agent
        df = pd.concat([df, temp_df], ignore_index=True)

# Reshape the DataFrame to a long format
df_melt = df.melt(id_vars="Agent", var_name='Opponent Algorithm', value_name='Return')
# Normalize the 'Return' values for each agent to fit between -1 and 1
# df_melt['Return'] = df_melt.groupby('Agent')['Return'].transform(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)
# Calculate mean return and standard error for each agent
df_mean = df_melt.groupby('Agent')['Return'].mean()
df_std_err = df_melt.groupby('Agent')['Return'].std() / np.sqrt(df_melt.groupby('Agent').size())
# Define a list of colors
colors = ['#5ec962', '#21918c', '#3b528b','#414487' ]  # blue, green, red
# Calculate the number of unique agents
num_agents = df_mean.shape[0]
# Repeat the color list to match the number of agents
colors = colors * (num_agents // len(colors)) + colors[:num_agents % len(colors)]

# Plot with different colors for each bar
# determine size of plot

ax = df_mean.plot(kind='bar', 
                  yerr=df_std_err, 
                  capsize=5, 
                  figsize=(6, 6), 
                  color=colors,
                  width = 0.8,
                  )
plt.xlabel('Opponent algorithm')
plt.xticks(rotation=45)
# plt.legend(title='Opponent', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Mean Return')
plt.title('Tiny dominoes: CFR vs Other Algorithms')
plt.tight_layout()


plt.savefig(results_dir + f"/{game_name}_cfr_vs_others.png")