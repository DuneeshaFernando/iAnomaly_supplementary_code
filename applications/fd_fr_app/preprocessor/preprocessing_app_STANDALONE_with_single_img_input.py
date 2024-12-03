# In order to make the standalone preprocessor look more similar to the actual preprocessor we modified it to take only one image as input. The first image is loaded at the start of the program.
import cv2
import numpy as np
from flask import Flask, request, jsonify
import urllib.request, json
import time
import base64

app = Flask(__name__)

def resize_image(image):
    resized_image = cv2.resize(image, (1080, 720))
    return resized_image

def grayscale_conversion(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def motion_detection(image_1, image_2):
    # Calculate absolute difference between the two frames
    frame_diff = cv2.absdiff(image_1, image_2)
    # Apply a threshold to extract significant differences
    _, thresholded_image = cv2.threshold(frame_diff, 30, 1, cv2.THRESH_BINARY)
    # Sum the values of thresholded_image, and obtain the ratio of 1s vs all pixels.
    sum_of_vals = np.sum(thresholded_image)
    count_of_vals = thresholded_image.size
    sum_to_count_ratio = sum_of_vals/count_of_vals
    # If it is greater than a hardcoded threshold, return a boolean value to main function
    if sum_to_count_ratio > 0.08:
        return True
    else:
        return False

@app.route("/process_image", methods=['POST'])
def image_preprocessing():
    # In this standalone version, the gray_image array of the first image is loaded when the program loads (refer to main).

    # st = time.time()

    image2_file = request.files['image2'].read()
    # Convert bytes to NumPy array
    img2_array = np.frombuffer(image2_file, np.uint8)
    # Decode NumPy array into image
    img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
    resized_image2 = resize_image(img2)

    # rt = time.time()
    # print(rt-st)

    gray_image2 = grayscale_conversion(resized_image2)

    # gt = time.time()
    # print(gt - rt)

    # If motion_detection returns True, invoke face detection by passing image2
    if motion_detection(gray_image,gray_image2):

        # mdt = time.time()
        # print(mdt - gt)

        # Call the FD microservice
        base_url = 'http://127.0.0.1:4000'
        path = "/detect_face"
        url = base_url + path

        # Serialize the ndarray to JSON -  The following part of code had to be improved to save time
        base64_encoded_data = base64.b64encode(resized_image2).decode('utf-8')
        json_data = json.dumps({'image': base64_encoded_data})
        # json_data = json.dumps(resized_image2.tolist())
        # jt = time.time()
        # print(jt - mdt)

        # Encode JSON data
        encoded_data = json_data.encode('utf-8')

        # Send POST request with JSON data
        req = urllib.request.Request(url, data=encoded_data, method='POST')
        req.add_header('Content-Type', 'application/json')

        # After this, we do not send the request. Instead, we mock the response sent from FD.
        dict = {"code": 0}

        # ft = time.time()
        # print(ft-st)

        return dict
    else:
        return jsonify(response="No face detected")

if __name__ == "__main__":
    # Step 1 : In order to prevent sending 2 images over the network and preprocessing both, I saved the gray_image array of the first image and loaded it.
    data_path = 'gray_image.npy' #'/data/gray_image.npy'
    gray_image = np.load(data_path)
    app.run(host='0.0.0.0', port=3000, debug=True)
