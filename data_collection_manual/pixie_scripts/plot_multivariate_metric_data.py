# The SN dataset does not have labels for anomaly detection. Therefore, in order to evaluate (and perform hyper parameter tuning for optimization), we need to have manual labels.
# For that purpose, we need to generate plots from the data
import pandas as pd
from matplotlib import pyplot as plt

# merge_df = pd.read_csv("../../sample_dataset_for_analysis/special_mdi_dataset_for_plotting.csv")
merge_df = pd.read_csv("../../sample_dataset_for_analysis/anomaly_data/preprocess_other_anomaly.csv")
# merge_df = merge_df.drop(["container_x"],axis=1)

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = plt.subplots(12)
ax1.plot(merge_df["total_disk_read_throughput"]) #total_disk_read_throughput
ax2.plot(merge_df["total_disk_write_throughput"]) #total_disk_write_throughput
ax3.plot(merge_df["rss"]) #rss
ax4.plot(merge_df["vsize"]) #vsize
ax5.plot(merge_df["cpu_usage"]) #cpu_usage
ax6.plot(merge_df["rx_bytes_per_ns"]) #rx_bytes_per_ns
ax7.plot(merge_df["tx_bytes_per_ns"]) #tx_bytes_per_ns
ax8.plot(merge_df["latency_p50"], marker='o', markersize=3) #latency_p50
ax9.plot(merge_df["latency_p90"], marker='o', markersize=3) #latency_p50
ax10.plot(merge_df["latency_p99"], marker='o', markersize=3) #latency_p50
ax11.plot(merge_df["request_throughput"], marker='o', markersize=3) #request_throughput
ax12.plot(merge_df["errors_per_ns"]) #errors_per_ns

plt.show()