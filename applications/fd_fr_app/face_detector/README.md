* This folder contains the code to run FD service.
* It runs at port 4000
* To run this application
  * as a Flask app (face_detection_app_FLASK_ONLY.py) and directly test, read from jmeter input and also set the base_url to localhost:5000
  * face_detection_app_STANDALONE.py file - is useful for normal data generation (including microservice property characterization, saturation point identification).
* Corresponding jmeter test script for local testing is `jmeter_tests/test_fd.jmx`.

* Docker image for http standalone version (face_detection_app_STANDALONE.py) - dtfernando/face_detect_standalone 
