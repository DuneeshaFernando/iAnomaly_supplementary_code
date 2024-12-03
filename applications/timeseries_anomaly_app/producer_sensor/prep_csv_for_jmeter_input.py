# Prepare the csv for jmeter input.
import pandas as pd
import random
import numpy as np

# Function to read CSV file and return it as a DataFrame. In the original function, is_anomaly attribute is dropped. In this function, we add a mock timestamp column
def read_csv(file_path):
    df = pd.read_csv(file_path)
    # Drop the is_anomaly attribute
    df = df.drop(['is_anomaly'], axis=1)
    # Add a mock timestamp column
    df['timestamp'] = 1615551440
    return df

# Main function
def main():
    csv_file_path = 'anomaly_data.csv' # service_chain_cpu_hog_step

    # Read CSV data while dropping is_anomaly attribute
    data = read_csv(csv_file_path)

    new_data = []

    # Randomly drop missing values.
    for index, row in data.iterrows():
        message = row.to_dict()
        # Get a list of keys from the dictionary
        keys = list(message.keys())
        keys = [e for e in keys if e not in ('datetime', 'timestamp')]
        if random.random() <= 0.05:
            # Drop few values from the row
            # Randomly determine the number of values to replace
            replace_count = random.randint(1, len(keys) // 2)
            # Randomly select keys to replace with N/A
            replace_keys = random.sample(keys, replace_count)
            # Replace the selected keys' values with N/A
            for key in replace_keys:
                message[key] = np.nan
        new_data.append(message)

    new_df = pd.DataFrame(new_data)
    new_df = new_df.fillna("null")

    # Write the csv
    new_df.to_csv('anomaly_data_jmeter_input_w_missing_vals_as_null.csv', index=False)

if __name__ == '__main__':
    main()