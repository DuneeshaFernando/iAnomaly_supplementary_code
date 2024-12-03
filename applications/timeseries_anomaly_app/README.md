* During local e2e run, as the first step, locally start the kafka container using `docker run -p 9092:9092 apache/kafka:3.7.1` command.
* Then start anomaly detector.
  * anomaly_detector_flask/if_ad_FLASK_LOCAL_RUN.py - suitable for debugging anomaly detector
  * anomaly_detector_gunicorn folder - contains the production-ready, thread-safe files. Command to execute is `gunicorn --config gunicorn_config.py if_ad_flask_app:app`
* Then start missing data imputer.
  * missing_data_imputer_flask/missing_data_impute_FLASK_LOCAL_RUN.py - contains the code in flask format. suitable for debugging.
  * missing_data_imputer_gunicorn folder - contains the production-ready, thread-safe files. Command to execute is `gunicorn --config gunicorn_config.py missing_data_impute_flask_app:app`
* Then, execute the producer_sensor/producer_sensor_KAFKA_LOCAL_RUN.py file.
* Finally, execute the subscriber_preprocessor/subscriber_preprocessor_KAFKA_LOCAL_RUN.py

* During standalone testing,
  * Independently test, AD and MDI.
    * AD standalone (flask) - 
      * code : anomaly_detector_flask/if_ad_FLASK_LOCAL_RUN_STANDALONE.py
      * Docker image : dtfernando/anomaly_detect_standalone_flask
      * jmeter script to test : timeseries_anomaly_app/jmeter_tests/if_ad_input.jmx (change port to 5000)
    * AD standalone (gunicorn) - 
      * code : anomaly_detector_gunicorn/if_ad_flask_app_STANDALONE.py. Note that, the command to execute is `gunicorn --config gunicorn_config.py if_ad_flask_app_STANDALONE:app`
      * Docker image : dtfernando/anomaly_detect_standalone_gunicorn
      * jmeter script to test : timeseries_anomaly_app/jmeter_tests/if_ad_input.jmx (change port to 8080)
    * MDI standalone (flask) -
      * code : missing_data_imputer_flask/missing_data_impute_FLASK_LOCAL_RUN_STANDALONE.py
      * Docker image : dtfernando/mdi_standalone_flask
      * jmeter script to test : timeseries_anomaly_app/jmeter_tests/mdi_input.jmx (change port to 5000)
    * MDI standalone (gunicorn) - 
      * code : missing_data_imputer_gunicorn/missing_data_impute_flask_app.py. Note that, the command to execute is `gunicorn --config gunicorn_config.py missing_data_impute_flask_app:app`
      * Docker image : dtfernando/mdi_standalone_gunicorn
      * jmeter script to test : timeseries_anomaly_app/jmeter_tests/mdi_input.jmx (change port to 8080)
  * Orchestrator needs to talk to AD and MDI. This requires a separate setup of preprocessor, AD and MDI.
    * A preprocessor (http-based) that receives requests from jmeter -
      * code : subscriber_preprocessor/subscriber_preprocessor_FLASK.py
      * Docker image : dtfernando/subscriber_preprocessor_flask
      * jmeter script to test : timeseries_anomaly_app/jmeter_tests/subscriber_preprocessor_input.jmx (it reads from the producer_sensor/anomaly_data_jmeter_input_w_missing_vals_as_null.csv with missing values. This file was prepared inside producer_sensor folder using a one-time code)
    * A missing data imputer capable of serving requests from the above preprocessor (use the dtfernando/mdi_gunicorn docker image from iAnomaly_toolkit repo)
    * An AD capable of serving requests from the above preprocessor (use the dtfernando/anomaly_detect_gunicorn docker image from iAnomaly_toolkit repo)
