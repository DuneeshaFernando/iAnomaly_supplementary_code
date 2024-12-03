# Contains the query to extract CPU usage, disk read/write, rss and vsize

import pxapi
import pandas as pd

# Define a PxL query with one output table.
PXL_SCRIPT = """
import px
ns_per_ms = 1000 * 1000
ns_per_s = 1000 * ns_per_ms
# Window size to use on time_ column for bucketing.
window_ns = px.DurationNanos(10 * ns_per_s)

df = px.DataFrame(table='http_events', start_time="-10m")
df.failure = df.resp_status >= 400

filter_out_conds = ((df.req_path != '/healthz') and (df.req_path != '/readyz')) and (df['remote_addr'] != '-')
df = df[filter_out_conds]

df.pod = df.ctx['pod']
df = df[df.pod == "default/fr-svc-deployment-6cdcb5cb79-lrhmm"]

# Filter only to inbound pod traffic (server-side).
# Don't include traffic initiated by this pod to an external location.
df = df[df.trace_role == 2]

df.container = df.ctx['container']
df.timestamp = px.bin(df.time_, window_ns)
df = df.groupby(['timestamp', 'container']).agg(
    error_rate_per_window=('failure', px.mean),
    throughput_total=('latency', px.count)
)

# Format the result of LET aggregates into proper scalar formats and
# time series.
df.request_throughput = df.throughput_total / window_ns
df.errors_per_ns = df.error_rate_per_window * df.request_throughput / px.DurationNanos(1)
df.time_ = df.timestamp
df = df.drop(['error_rate_per_window','throughput_total','timestamp'])
px.display(df, 'throughput_error_table')
"""

# Create a Pixie client.
px_client = pxapi.Client(token="px-api-3a43ec3e-5d5c-47b4-9c33-29f22335cd46") # Use px api-key create to identify API key

# Connect to cluster.
conn = px_client.connect_to_cluster("9ee3ab1a-dcbc-47e7-9859-2397c9edd35e") # Use px get viziers to identify cluster ID

# Execute the PxL script.
script = conn.prepare_script(PXL_SCRIPT)

# # print results
# for row in script.results("throughput_error_table"):
#     print(row)

d = {'container':[], 'request_throughput':[], 'errors_per_ns':[], 'time_':[]}

# Print the table output.
for row in script.results("throughput_error_table"):
    # Populate final_df with values from the row
    for k in d.keys():
        d[k].append(row[k])

final_df = pd.DataFrame(data=d)
final_df = final_df.sort_values('time_')

final_df.to_csv('collected_metric_dfs/throughput_error_timeseries.csv', index=False)

print("debug")
