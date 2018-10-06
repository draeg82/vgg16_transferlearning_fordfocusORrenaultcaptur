from keras.models import model_from_json
import matplotlib.pyplot as plt
import cv2
from random import shuffle

import numpy as np

# IMAGE META
IMG_CHANNELS = 3
IMG_ROWS = 224
IMG_COLS = 224

# Load model and weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model = loaded_model

files = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013']
shuffled_files = shuffle(files)

for file in files:
    print(file)
    image_path = 'data/test/%s.jpg' % file

    img = cv2.imread(image_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(img, (IMG_ROWS, IMG_COLS))
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    im = np.expand_dims(im, axis=0)
    out = model.predict(im)

    if np.argmax(out) == 0:
        confidence = out.item(0)
        confidence = str(round(confidence * 100, 1))
        title = 'Class: Captur   Confidence: %s' % confidence + '%'
        plt.title('Class: Captur   Confidence: %s' % confidence + '%')
    elif np.argmax(out) == 1:
        confidence = out.item(1)
        confidence = str(round(confidence * 100, 1))
        title = 'Class: Focus   Confidence: %s' % confidence + '%'
        plt.title('Class: Focus   Confidence: %s' % confidence + '%')
    plt.show()


