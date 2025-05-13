# Нейронные сети:
# — свёрточные (конволюционные) нейронные сети (CNN) — компьютерное зрение, классификация изображений
# — рекуррентные нейронные сети (RNN) — распознавания рукописного текста, обработка естественного языка
# — генеративные состязательные сети (GAN) — создание художественных, музыкальных произведений
# — многослойный перцептрон — простейший тип нейронных сетей

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# img_path = "./data/luna.jpg/"
# i don't have your images so here are some of train directory we use in next file and my cat <3
# img_path = '/home/dasha/Загрузки/dogs-vs-cats-redux-kernels-edition/train/dog.1.jpg'
img_path='mycat.png'

img = image.load_img(img_path, target_size = (224,224))
plt.imshow(img)

plt.show()
import numpy as np

img_array = image.img_to_array(img)
print(img_array.shape)

img_batch = np.expand_dims(img_array, axis=0)
print(img_batch.shape)

from tensorflow.keras.applications.resnet50 import preprocess_input

img_processed = preprocess_input(img_batch)

from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50()
prediction = model.predict(img_processed)

from tensorflow.keras.applications.resnet50 import decode_predictions
print(decode_predictions(prediction, top=5)[0])

# plt.imshow(img)
# plt.show()