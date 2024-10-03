import json
import time
from functools import wraps

import tensorflow as tf

from src.logger.log import logger
from src.models.model_manager_config import ModelManagerConfig
from src.commons import OffloadingDataFiles

def track_inference_time(func):
    """
    This decorator is used to track the execution time of a function.
    :param func: the function to be decorated
    :return: the decorated function
    """

    @wraps(func)
    def wrapper(self, layer_id: int, *args, **kwargs) -> object:
        # Start the timer
        start_time = time.time()
        # Execute the original function (predict_single_layer)
        result = func(self, layer_id, *args, **kwargs)
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        # Store the elapsed time in the inference_times dictionary
        self.inference_times[layer_id] = elapsed_time
        logger.debug(f"Edge Inference for layer [{layer_id}] took {elapsed_time:.4f} seconds")
        return result

    return wrapper


class ModelManager:
    """This class is used to manage the model and its layers.

    Args:
        save_path: The path to save the model.
        model_path: The path to the model.

    Attributes:
        save_path: The path to save the model.
        model_path: The path to the model.
        num_layers: The number of layers in the model.
        model: The model.
        inference_times: A dictionary to store the inference times for each layer.
    """

    def __init__(self, save_path: str = ModelManagerConfig.SAVE_PATH, model_path: str = ModelManagerConfig.MODEL_PATH):
        self.save_path = save_path
        self.model_path = model_path
        self.num_layers = None
        self.model = None
        # dictionary to store inference times for each layer
        self.inference_times = {}

    def load_model(self, model_path: str = ModelManagerConfig.MODEL_PATH):
        """Load the model from the given path.
        Args:
            model_path: The path to the model.
        Returns:
            None
        """
        logger.debug(f"Loading model from path: {model_path}")
        try:
            self.model_path = model_path
            self.model = tf.keras.models.load_model(model_path)
            self.num_layers = len(self.model.layers)
        except Exception as e:
            print(f"Error loading model: {e}")
            logger.error(f"Failed to load model: {e}")

    def get_model_layer(self, layer_id: int) -> tf.keras.layers.Layer:
        """Get the layer with the given id from the model.
        Args:
            layer_id: The id of the layer.
        Returns:
            The layer with the given id.
        """
        return self.model.layers[layer_id]

    @staticmethod
    def get_layer_size_in_bytes(layer: tf.keras.layers.Layer, layer_output: tf.Tensor) -> int:
        """Calculate the size of a Keras layer's weights in bytes.
        Args:
            layer: A Keras layer.
            layer_output: The output of the layer.
        Returns:
            The size of the layer in bytes.
        """

        """
        total_size_in_bytes = 0
        # Iterate through the weights of the layer
        for weight in layer.weights:
            # Get the shape of the weight tensor
            weight_shape = weight.shape
            # Get the dtype of the weights (as a string, like 'float32')
            dtype = weight.dtype
            # Use np.dtype directly since dtype is a string (e.g., 'float32')
            size_per_element = np.dtype(dtype).itemsize
            # Calculate the total number of elements
            num_elements = np.prod(weight_shape)
            # Total size in bytes for this weight
            size_in_bytes = num_elements * size_per_element
            total_size_in_bytes += size_in_bytes
        """
        # get the dtype of the output tensor
        dtype = layer_output.dtype
        # calculate the size in size_in_bytes and convert to Python scalar
        size_in_bytes = tf.reduce_prod(layer_output.shape) * tf.constant(dtype.itemsize)
        size_in_bytes = size_in_bytes.numpy()
        return size_in_bytes

    @track_inference_time
    def predict_single_layer(self, layer_id: int, layer_input_data: object) -> object:
        """Predict the output of a single layer.
        Args:
            layer_id: The id of the layer.
            layer_input_data: The input data to the layer.
        Returns:
            The output of the layer.
        """
        logger.debug(f"Making a prediction for layer [{layer_id}]")
        layer = self.get_model_layer(layer_id)
        # create an intermediate model with the current layer
        intermediate_model = tf.keras.Model(inputs=layer.input, outputs=layer.output)
        layer_output = intermediate_model.predict(layer_input_data)
        return layer_output

    def save_inference_times(self, save_path: str | None = None):
        """Save the inference times to a JSON file.
        Args:
            save_path: The path to save the inference times.
        Returns:
            None
        """
        if save_path is not None:
            self.save_path = save_path
        self.save_path = self.save_path[:-1] if self.save_path[-1] == "/" else self.save_path
        inference_times = self.inference_times
        with open(f"{self.save_path}/{OffloadingDataFiles.data_file_path_edge}.json", "w") as f:
            json.dump(inference_times, f, indent=4)
        logger.debug(f"Inference times saved")
