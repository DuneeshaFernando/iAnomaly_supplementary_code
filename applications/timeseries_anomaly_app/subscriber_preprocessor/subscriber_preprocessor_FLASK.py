from flask import Flask, request, jsonify
import pandas as pd
import random
import requests
import numpy as np
import os

# Flask app initialization
app = Flask(__name__)

class ReservoirSampler:
    def __init__(self, size):
        self.size = size
        self.reservoir = []
        self.count = 0

    def add(self, item):
        self.count += 1
        if len(self.reservoir) < self.size:
            self.reservoir.append(item)
        else:
            # Randomly replace an item in the reservoir with the new item
            s = random.randint(0, self.count - 1)
            if s < self.size:
                self.reservoir[s] = item

    def get_mu_sigma(self):
        # Form a dataframe from reservoir
        reservoir_df = pd.DataFrame(self.reservoir)
        # Obtain mu and sigma from each attribute of reservoir_df
        reservoir_means = reservoir_df[['in_avg_response_time', 'in_throughput', 'in_progress_requests', 'http_error_count', 'ballerina_error_count', 'cpu', 'memory', 'cpuPercentage', 'memoryPercentage']].mean()
        reservoir_stds = reservoir_df[['in_avg_response_time', 'in_throughput', 'in_progress_requests', 'http_error_count', 'ballerina_error_count', 'cpu', 'memory', 'cpuPercentage', 'memoryPercentage']].std()
        # Merge reservoir_means and reservoir_stds, and form mean_std_dict
        mean_std_dict = {col: {'mean': reservoir_means[col], 'std': reservoir_stds[col]} for col in reservoir_means.index}
        return mean_std_dict

reservoir_size = 5

# Create a reservoir sampler
reservoir_sampler = ReservoirSampler(reservoir_size)

window_buffer = []
window_size = 10

ad_base_url = os.environ.get('AD_URL')
ad_path = "/detect_anomalies"
ad_url = ad_base_url + ad_path
# ad_url = 'http://100.88.28.171:5000/detect_anomalies' # Replace url based on AD (gunicorn : 8080, flask : 5000)

mdi_base_url = os.environ.get('MDI_URL')
mdi_path = "/impute_missing_data"
imputation_url = mdi_base_url + mdi_path
# imputation_url = 'http://100.88.28.171:5002/impute_missing_data'

# Function that mocks consume_from_kafka function
@app.route('/process_event', methods=['POST'])
def process_event():
    event_data = request.json

    # Add the event data to the reservoir sampler
    reservoir_sampler.add(event_data)

    # Add the event data to the window buffer
    window_buffer.append(event_data)

    if len(window_buffer) >= window_size:
        # Form a dataframe by using the dictionaries in the window buffer
        df = pd.DataFrame(window_buffer)
        # Drop datetime and timestamp from df
        df = df.drop(['datetime','timestamp'], axis=1)
        # Check for missing values in df. If there are any missing values, send that for imputation
        if df.isna().sum().sum()>0:
            # Send a POST request to the Imputation microservice
            response = requests.post(imputation_url, json=df.to_json(orient='records'))

            # Check if the imputation was successful
            if response.status_code == 200:
                imputed_data = response.json()
                df = pd.read_json(imputed_data)
            else:
                print('Failed to impute missing data:', response.status_code)

        mean_std_dict = reservoir_sampler.get_mu_sigma()

        # Standardize the df
        for col in mean_std_dict.keys():
            mean = mean_std_dict[col]['mean']
            std = mean_std_dict[col]['std']  # + 2.22507E-308
            df[col] = (df[col] - mean) / std

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN values with 0
        df = df.fillna(0)

        # Send a POST request to the AD Flask app
        response = requests.post(ad_url, json=df.to_json(orient='records'))

        # Clear the window buffer after processing
        window_buffer.clear()

        # Check the status code of the response
        if response.status_code == 200:
            print('Response:', response.json())
        else:
            print('Failed to retrieve data:', response.status_code)
        return {'Response': response.json()}

    # Return a message indicating that the buffer hasn't reached the window size yet
    else:
        return jsonify({'message': f'Buffer size: {len(window_buffer)}. Waiting for more events.'}), 200


# Main function
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)



