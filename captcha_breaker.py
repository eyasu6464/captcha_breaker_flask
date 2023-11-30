# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2
import flask
import pandas as pd
from PIL import Image

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
# import pandas as pd

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    processer = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model =  CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        if cls.processer == None:
            cls.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return cls.model, cls.processor

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        objects=["bicycle", "chiar","window", "ball", "ship","tv screen","bag","cat", "phone", "cow", "hammer", "fox", "dog", "umbrella","Watering can"]
        
        clf, clf_processer = cls.get_model()
        # input
        img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        # Read the image
        # img = cv2.imread('/content/drafar05.jpg')

        # Split the image into five equal parts horizontally and vertically
        h, w = img.shape[:2]
        img1 = img[int(h // (30)):int(h // (2.6)), int(w // (2.42)):int(w // (1.71))]
        img2 = img[int(h // (1.8)):int(h // (1.05)), int(w // (8.5)):int(w // (3.3))]
        img3 = img[int(h // (1.8)):int(h // (1.05)), int(w // (3.3)): int(w // (2.0))]
        img4 = img[int(h // (1.8)):int(h // (1.05)), int(w // (2.0)):  int(w // (1.445))]
        img5 = img[int(h // (1.8)):int(h // (1.05)),  int(w // (1.445)):int(w // (1.13))]

        query_image=Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        candidate_images=[
            Image.fromarray(img2),
            Image.fromarray(img3),
            Image.fromarray(img4),
            Image.fromarray(img5)
        ]
        best_image = cls.solve_puzzle(query_image, candidate_images, clf, clf_processer)
        return best_image
    
    @classmethod
    def solve_puzzle(cls,query_imagel, candidate_images, model, processor):
        # Preprocess the query image and the candidate images using the processor
        objects=["bicycle", "chiar","window", "ball", "ship","tv screen","bag","cat", "phone", "cow", "hammer", "fox", "dog", "umbrella","Watering can"]

        query_image = processor(text=objects,images=query_imagel, return_tensors="pt", padding=True)
        query_caption = model(**query_image)
        probs = query_caption.logits_per_image.softmax(dim=1)
        index= probs[0].argmax()
        # print(probs[0])
        print(objects[index])
        candidate_images=[
            processor(text=objects,images=c_image, return_tensors="pt", padding=True) for c_image in candidate_images
        ]

        query_caption = model(**query_image)
        candidate_captions = [
            float(model(**candidate_image).logits_per_image.softmax(dim=1)[0][index]) for candidate_image in candidate_images
        ]
        candidate_captions=[float(i) for i in candidate_captions]
        max_val = max(candidate_captions)
        # print(max_val)
        max_index = candidate_captions.index(max_val)
        max_index +=1
        # print(max_index)
        return f"{max_index}"



app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""

    status = 200
    return flask.Response(response="\n", status=status, mimetype="application/json")

# @app.route("/invocations", methods=["POST"])
# def transformation():
#     """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
#     it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
#     just means one prediction per line, since there's a single column.
#     """

#     status = 200
#     return flask.Response(response="blalbalbal", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    if not flask.request.mimetype.startswith('multipart/form-data'):
        return flask.Response(response="This predictor only supports multipart/form-data", status=415, mimetype="text/plain")

    # Check if the post request has the file part
    if "file" not in flask.request.files:
        return flask.Response(response="No file part in the request", status=400, mimetype="text/plain")

    file = flask.request.files["file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        return flask.Response(response="No selected file", status=400, mimetype="text/plain")

    if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Read the image via file.stream
        img = cv2.imdecode(np.frombuffer(file.stream.read(), np.uint8), 1)
        # cv2.imwrite("image.png", img)
    else:
        return flask.Response(response="This predictor only supports png, jpg and jpeg data", status=415, mimetype="text/plain")

    print("Invoked with an image")
    # Convert from CSV to pandas
    
    # Do the prediction
    predictions = ScoringService.predict(img)
    response_data = {
        "data": predictions
    }

    # Convert the dictionary to a JSON string
    response_json = json.dumps(response_data)

    # Return the JSON response
    return flask.Response(response=response_json, status=200, mimetype="application/json")
app.run(debug=True)