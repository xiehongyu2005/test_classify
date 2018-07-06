from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import io
from PIL import Image


def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

def classify(img):
    image = Image.open(io.BytesIO(img))
    image = prepare_image(image, target=(224, 224))
    model = ResNet50(weights="imagenet")
    preds = model.predict(image)
    results = imagenet_utils.decode_predictions(preds)
    data = []

    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": round(float(prob),2)}
        data.append(r)

    return data