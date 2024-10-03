import json

from pytest import fixture

from tests.commons import TestSamples


@fixture
def device_offloading_data():
    with open(TestSamples.data_file_path_device, 'r') as file:
        device_inference_times = json.load(file)
    device_inference_times = list({k: v for k, v in device_inference_times.items()}.values())
    return device_inference_times


@fixture
def edge_offloading_data():
    with open(TestSamples.data_file_path_edge, 'r') as file:
        edge_inference_times = json.load(file)
    edge_inference_times = list({k: v for k, v in edge_inference_times.items()}.values())
    return edge_inference_times


@fixture
def layers_sizes_offloading_data():
    with open(TestSamples.data_file_path_sizes, 'r') as file:
        layers_sizes = json.load(file)
    layers_sizes = list({k: v for k, v in layers_sizes.items()}.values())
    return layers_sizes
