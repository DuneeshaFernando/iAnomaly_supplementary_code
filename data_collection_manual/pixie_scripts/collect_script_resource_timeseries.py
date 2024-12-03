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
df = px.DataFrame(table='process_stats', start_time="-10m")
df = df[df.ctx['pod'] == "default/fr-svc-deployment-6cdcb5cb79-lrhmm"]
df.timestamp = px.bin(df.time_, window_ns)
df.container = df.ctx['container_name']

# First calculate CPU usage by process (UPID) in each k8s_object
# over all windows.
df = df.groupby(['upid', 'container', 'timestamp']).agg(
    rss=('rss_bytes', px.mean),
    vsize=('vsize_bytes', px.mean),
    # The fields below are counters, so we take the min and the max to subtract them.
    cpu_utime_ns_max=('cpu_utime_ns', px.max),
    cpu_utime_ns_min=('cpu_utime_ns', px.min),
    cpu_ktime_ns_max=('cpu_ktime_ns', px.max),
    cpu_ktime_ns_min=('cpu_ktime_ns', px.min),
    rchar_bytes_max=('rchar_bytes', px.max),
    rchar_bytes_min=('rchar_bytes', px.min),
    wchar_bytes_max=('wchar_bytes', px.max),
    wchar_bytes_min=('wchar_bytes', px.min),
)


# Next calculate cpu usage and memory stats per window.
df.cpu_utime_ns = df.cpu_utime_ns_max - df.cpu_utime_ns_min
df.cpu_ktime_ns = df.cpu_ktime_ns_max - df.cpu_ktime_ns_min
df.total_disk_read_throughput = (df.rchar_bytes_max - df.rchar_bytes_min) / window_ns
df.total_disk_write_throughput = (df.wchar_bytes_max - df.wchar_bytes_min) / window_ns

# Then aggregate process individual process metrics.
df = df.groupby(['timestamp', 'container']).agg(
    cpu_ktime_ns=('cpu_ktime_ns', px.sum),
    cpu_utime_ns=('cpu_utime_ns', px.sum),
    total_disk_read_throughput=('total_disk_read_throughput', px.sum),
    total_disk_write_throughput=('total_disk_write_throughput', px.sum),
    rss=('rss', px.sum),
    vsize=('vsize', px.sum),
)

# Finally, calculate total (kernel + user time)  percentage used over window.
df.cpu_usage = px.Percent((df.cpu_ktime_ns + df.cpu_utime_ns) / window_ns)
df.time_ = df.timestamp
df = df.drop(['cpu_ktime_ns', 'cpu_utime_ns', 'timestamp'])
px.display(df, 'resource_table')
"""

# Create a Pixie client.
px_client = pxapi.Client(token="px-api-3a43ec3e-5d5c-47b4-9c33-29f22335cd46") # Use px api-key create to identify API key

# Connect to cluster.
conn = px_client.connect_to_cluster("9ee3ab1a-dcbc-47e7-9859-2397c9edd35e") # Use px get viziers to identify cluster ID

# Execute the PxL script.
script = conn.prepare_script(PXL_SCRIPT)

d = {'container':[], 'total_disk_read_throughput':[], 'total_disk_write_throughput':[], 'rss':[], 'vsize':[], 'cpu_usage':[], 'time_':[]}

# Print the table output.
for row in script.results("resource_table"):
    # Populate final_df with values from the row
    for k in d.keys():
        d[k].append(row[k])

final_df = pd.DataFrame(data=d)
final_df = final_df.sort_values('time_')

final_df.to_csv('collected_metric_dfs/resource_timeseries.csv', index=False)

print("debug")
