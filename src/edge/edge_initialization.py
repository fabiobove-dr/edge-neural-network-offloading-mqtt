import json

import numpy as np

from src.commons import OffloadingDataFiles
from src.models.model_manager import ModelManager

if __name__ == "__main__":
    # original array (grayscale image with shape (1, 10, 10))
    image_array = np.array([[
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 0, 0, 0, 255, 255, 255, 255, 255],
        [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
        [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
        [255, 255, 0, 0, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 0, 0, 255, 255, 255],
        [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
        [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    ]])

    # 1. expand the array to have 3 channels by repeating along the third axis
    image_array_3_channels = np.repeat(image_array[..., np.newaxis], 3, axis=-1)

    # 2. convert the image to float32 to match the expected input type of the model
    image_array_3_channels = image_array_3_channels.astype(np.float32)

    # 3. normalize the image (optional, but often required by models)
    image_array_3_channels /= 255.0

    # check the shape and dtype
    print(image_array_3_channels.shape)  # Now the shape should be (1, 10, 10, 3)
    print(image_array_3_channels.dtype)  # Should print float32

    # convert the NumPy array to a string representation for writing to a file
    image_str = np.array2string(image_array_3_channels)

    # load the model and make predictions
    model_manager = ModelManager()
    model_manager.load_model()

    # set the layers to use
    predictions = []
    layer_sizes = {}
    start_layer_index = 0

    # set end_layer_index to the number of layers if not provided
    end_layer_index = len(model_manager.model.layers)

    # set the layers to use
    layers_to_use = model_manager.model.layers[start_layer_index:end_layer_index]

    # loop through the layers and make predictions
    input_data = image_array_3_channels
    for layer_index, layer in enumerate(layers_to_use, start=start_layer_index):
        prediction_data = input_data if layer_index == start_layer_index else predictions[-1]
        prediction = model_manager.predict_single_layer(layer_index, prediction_data)
        layer_sizes[layer_index] = float(model_manager.get_layer_size_in_bytes(layer, prediction))
        predictions.append(prediction)

    # save the inference times to a file
    model_manager.save_inference_times()

    # save the layer sizes to a file
    with open(f"../{OffloadingDataFiles.data_file_path_sizes}.json", "w") as f:
        json.dump(layer_sizes, f, indent=4)
