This folder contains several forms of the preprocessor microservice.
* preprocessing_app.py - is the Flask form of the preprocessor microservice
  * You can test it along with the other 2 Flask apps for FD and FR.
  * It runs at port 3000
  * Corresponding jmeter test script for local testing is `jmeter_tests/test_preprocessor.jmx`
* rtsp_preprocessor_LOCAL_TESTING_ONLY.py - is the same as rtsp_preprocessor.py (which is the dockerizable version in the iAnomaly toolkit main repo), but useful for local testing.
* preprocessing_app_STANDALONE_with_single_img_input.py - is useful for normal data generation (including microservice property characterization, saturation point identification). 

* Docker image for http standalone version (preprocessing_app_STANDALONE_with_single_img_input.py) - dtfernando/preprocess_standalone_single_img_inp
