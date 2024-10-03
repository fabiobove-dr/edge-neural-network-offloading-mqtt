import os

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from model_manager_config import ModelManagerConfig


def init_folders(root_folder: str) -> None:
    os.makedirs(f"{root_folder}/layers/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/keras/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/tflite/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/h/", exist_ok=True)


def to_tflite(keras_model: Model, save: bool, save_dir: str, name: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    if save:
        with open(f"{save_dir}/{name}.tflite", 'wb') as f:
            f.write(tflite_model)
    return tflite_model


def save_keras(name: str, model: Model, dir_path: str) -> None:
    model.save(f'{dir_path}/{name}.keras')


def load_keras(name: str, dir_path: str) -> Model:
    return tf.keras.models.load_model(f'{dir_path}/{name}.keras')


def create_keras_submodels(save_dir: str, model: Model) -> dict:
    submodels = {}
    # start with the input tensor
    input_tensor = Input(shape=model.input_shape[1:])
    x = input_tensor
    # iterate through layers and create submodels
    for i, layer in enumerate(model.layers):
        # pass the input tensor through each layer sequentially
        x = layer(x)
        # create the submodel from the input tensor to the current layer's output
        submodel = Model(inputs=input_tensor, outputs=x)
        submodels[layer.name] = submodel
        # save each submodel to a file
        submodel.save(f'{save_dir}/submodel_{i}.keras')
    return submodels


def build_resnet_from_scratch(img_height=10, img_width=10, num_classes=5) -> Model:
    # inputs = layers.Input(shape=(img_height, img_width, 3))
    resnet_model = tf.keras.models.Sequential()
    # initial Conv Layer
    resnet_model.add(
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(img_height, img_width, 3)))
    resnet_model.add(layers.BatchNormalization())
    resnet_model.add(layers.ReLU())
    resnet_model.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
    resnet_model.add(layers.Dense(num_classes, activation='softmax'))
    return resnet_model


if __name__ == "__main__":

    # initialize folders
    main_folder = "test/" + ModelManagerConfig.MODEL_DIR_PATH
    print(f"main_folder: {main_folder}")
    init_folders(main_folder)

    # build 'keras' model and store it
    print("building keras model ...")
    model = build_resnet_from_scratch(img_height=ModelManagerConfig.IMAGE_SIZE, img_width=ModelManagerConfig.IMAGE_SIZE)
    save_keras(name="resnet_model", model=model, dir_path=main_folder)

    # load the model
    model = load_keras(name="resnet_model", dir_path=main_folder)

    # creates and save submodels '.tflite'
    print("creating submodels ...")
    submodels = create_keras_submodels(model=model, save_dir=f"{main_folder}/layers/keras")

    # creates and save submodels '.h'
    for layer_index, item in enumerate(submodels.items()):
        model_name, model = item
        # convert the model content to a C array format
        print(f"created [tflite] submodel for layer: {layer_index}")
        tflite_bytes = to_tflite(model, save=True, save_dir=f"{main_folder}/layers/tflite",
                                 name=f"submodel_{layer_index}")

        # convert the model content to a C array format
        model_array = ", ".join([str(b) for b in tflite_bytes])

        # write the C header file
        print(f"created [h] submodel for layer: {layer_index}")
        with open(f'{main_folder}/layers/h/layer_{layer_index}.h', 'w') as header_file:
            header_file.write('#pragma once\n\n')
            header_file.write('#include <cstdint>\n\n')
            header_file.write('const uint8_t layer_' + str(layer_index - 1) + '[] = {\n')
            header_file.write(model_array)
            header_file.write('\n};\n')

    print(model.summary())
