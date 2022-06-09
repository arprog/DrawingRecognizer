import cv2 as cv
import numpy as np
from PIL import Image
from tensorflow import keras
from flask import Flask, request
from skimage.filters import threshold_otsu
from tensorflow_addons.metrics import F1Score

app = Flask("api")
model = keras.models.load_model('./model.h5', custom_objects={"Addons>F1Score" : F1Score(num_classes=10)})

types_drawings = {'0': "Avião",
                  '1': "Banana",
                  '2': "Abelha",
                  '3': "Copo de café",
                  '4': "Caranguejo",
                  '5': "Guitarra",
                  '6': "Hambúrguer",
                  '7': "Coelho",
                  '8': "Caminhão",
                  '9': "Guarda-chuva"}

def response(status, message, http_code):
    return { "status" : status, "message" : message }, http_code

@app.route("/analyze-image", methods=["POST"])
def post():
    try:
        image_request = request.files['image']
        image_pil = Image.open(image_request)
        image = cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR)
        image = cv.resize(image,(130,130))

        gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        thresh = threshold_otsu(gray_image)
        black_white_image = gray_image > thresh

        analysis_result = model.predict(np.array([black_white_image]))
        analysis_result = analysis_result[0].tolist()
        
        greater_value = max(analysis_result)
        index_greater_value = analysis_result.index(greater_value)
        identified_drawing = types_drawings[str(index_greater_value)]
        return response(True, identified_drawing, 200)
    except:
        return response(False, identified_drawing, 500)

app.run()