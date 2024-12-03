# Contains the query to extract CPU usage, disk read/write, rss and vsize

import pxapi
import pandas as pd

pod_name = "default/fr-svc-deployment-6cdcb5cb79-lrhmm"

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
df = df[df.pod == "{0}"]

# Filter only to inbound pod traffic (server-side).
# Don't include traffic initiated by this pod to an external location.
df = df[df.trace_role == 2]

df.timestamp = px.bin(df.time_, window_ns)
df = df.groupby(['timestamp']).agg(
    latency_quantiles=('latency', px.quantiles)
)
df.latency_p50 = px.DurationNanos(px.floor(px.pluck_float64(df.latency_quantiles, 'p50')))
df.latency_p90 = px.DurationNanos(px.floor(px.pluck_float64(df.latency_quantiles, 'p90')))
df.latency_p99 = px.DurationNanos(px.floor(px.pluck_float64(df.latency_quantiles, 'p99')))
df.time_ = df.timestamp
df = df.drop(['latency_quantiles', 'timestamp'])
px.display(df, 'latency_table')
""".format(pod_name)

# Create a Pixie client.
px_client = pxapi.Client(token="px-api-3a43ec3e-5d5c-47b4-9c33-29f22335cd46") # Use px api-key create to identify API key

# Connect to cluster.
conn = px_client.connect_to_cluster("9ee3ab1a-dcbc-47e7-9859-2397c9edd35e") # Use px get viziers to identify cluster ID

# Execute the PxL script.
script = conn.prepare_script(PXL_SCRIPT)
print(PXL_SCRIPT)

# # print results
# for row in script.results("latency_table"):
#     print(row)

d = {'latency_p50':[],'latency_p90':[],'latency_p99':[],'time_':[]}

# Print the table output.
for row in script.results("latency_table"):
    # Populate final_df with values from the row
    for k in d.keys():
        d[k].append(row[k])

final_df = pd.DataFrame(data=d)
final_df = final_df.sort_values('time_')

final_df.to_csv("collected_metric_dfs/latency_timeseries.csv", index=False)

print("debug")
