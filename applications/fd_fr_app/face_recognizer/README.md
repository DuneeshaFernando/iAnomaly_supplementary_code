* This folder contains the code to run FR service. 
* It runs at port 5000
* To run this application 
  * as a Flask app (face_recognition_app_FLASK_ONLY.py) - To test locally. identical to the dockerizing version in terms of the functionality. i.e. it accepts the input from face detector.
  * face_recognition_app_STANDALONE.py file - is useful for normal data generation (including microservice property characterization, saturation point identification). It works without mysql database.
* Corresponding jmeter test script for local testing is `jmeter_tests/test_fr.jmx`.

* Docker image for http standalone version (face_recognition_app_STANDALONE.py) - dtfernando/face_recog_standalone
