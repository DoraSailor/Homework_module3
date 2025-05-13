import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.preprocessing import image

# took some random from dogs-vs-cats-redux-kernels-edition
TRAIN_DATA_DIR = "train"
VALIDATION_DATA_DIR = "val"
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

train_datagen = image.ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2
)

val_datagen = image.ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 12345,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 12345,
    class_mode="categorical"
)

from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

def model_maker():
    base_model = MobileNet(include_top = False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False

        input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        custom_model = base_model(input)
        custom_model = GlobalAveragePooling2D()(custom_model)
        custom_model = Dense(64, activation="relu")(custom_model)
        custom_model = Dropout(0.5)(custom_model)
        prediction = Dense(NUM_CLASSES, activation="softmax")(custom_model)
        return Model(inputs = input, outputs = prediction)

from tensorflow.keras.optimizers import Adam

model = model_maker()
model.compile(loss = "categorical_crossentropy",optimizer=Adam(), metrics=["acc"])

import math
num_steps = math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10, # Шаг обучения
    validation_data=val_generator,
    validation_steps=num_steps,
)

print(val_generator.class_indices)

model.save("./data/model.h5")
