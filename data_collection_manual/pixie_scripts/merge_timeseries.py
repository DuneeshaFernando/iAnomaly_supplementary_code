# Merge all collected time series in the collected_metric_dfs folder based on time_ column
import pandas as pd

folder_name = "collected_metric_dfs"

resource_df = pd.read_csv(folder_name+"/resource_timeseries.csv")
network_df = pd.read_csv(folder_name+"/network_timeseries.csv")
latency_df = pd.read_csv(folder_name+"/latency_timeseries.csv")
throughput_error_df = pd.read_csv(folder_name+"/throughput_error_timeseries.csv")

# Merge the dataframes
merged_df = resource_df.merge(network_df, on='time_', how='outer') \
               .merge(latency_df, on='time_', how='outer') \
               .merge(throughput_error_df, on='time_', how='outer')

# Fill missing values with a specified value (e.g., 0 or NaN)
# merged_df = merged_df.fillna('NaN')
merged_df = merged_df.drop(["container_y"],axis=1)

merged_df.to_csv('sample_dataset/fr_usersurge_anomaly.csv', index=False)

print("debug")