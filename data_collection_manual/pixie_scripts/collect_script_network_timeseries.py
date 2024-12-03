# Contains the query to extract rx_bytes_per_ns and tx_bytes_per_ns

import pxapi
import pandas as pd

# Define a PxL query with one output table.
PXL_SCRIPT = """
import px
ns_per_ms = 1000 * 1000
ns_per_s = 1000 * ns_per_ms
# Window size to use on time_ column for bucketing.
window_ns = px.DurationNanos(10 * ns_per_s)
df = px.DataFrame(table='network_stats', start_time="-10m")
df = df[df.ctx['pod'] == "default/fr-svc-deployment-6cdcb5cb79-lrhmm"]
df.timestamp = px.bin(df.time_, window_ns)

# First calculate network usage by node over all windows.
# Data is sharded by Pod in network_stats.
df = df.groupby(['timestamp', 'pod_id']).agg(
    rx_bytes_end=('rx_bytes', px.max),
    rx_bytes_start=('rx_bytes', px.min),
    tx_bytes_end=('tx_bytes', px.max),
    tx_bytes_start=('tx_bytes', px.min),
)

# Calculate the network statistics rate over the window.
# We subtract the counter value at the beginning ('_start')
# from the value at the end ('_end').
df.rx_bytes_per_ns = (df.rx_bytes_end - df.rx_bytes_start) / window_ns
df.tx_bytes_per_ns = (df.tx_bytes_end - df.tx_bytes_start) / window_ns

# Add up the network values per node.
df = df.groupby(['timestamp']).agg(
    rx_bytes_per_ns=('rx_bytes_per_ns', px.sum),
    tx_bytes_per_ns=('tx_bytes_per_ns', px.sum),
)
df.time_ = df.timestamp
df = df.drop(['timestamp'])
px.display(df, 'network_table')
"""

# Create a Pixie client.
px_client = pxapi.Client(token="px-api-3a43ec3e-5d5c-47b4-9c33-29f22335cd46") # Use px api-key create to identify API key

# Connect to cluster.
conn = px_client.connect_to_cluster("9ee3ab1a-dcbc-47e7-9859-2397c9edd35e") # Use px get viziers to identify cluster ID

# Execute the PxL script.
script = conn.prepare_script(PXL_SCRIPT)

# # print results
# for row in script.results("network_table"):
#     print(row)

d = {'rx_bytes_per_ns':[], 'tx_bytes_per_ns':[], 'time_':[]}

# Print the table output.
for row in script.results("network_table"):
    # Populate final_df with values from the row
    for k in d.keys():
        d[k].append(row[k])

final_df = pd.DataFrame(data=d)
final_df = final_df.sort_values('time_')

final_df.to_csv('collected_metric_dfs/network_timeseries.csv', index=False)

print("debug")
