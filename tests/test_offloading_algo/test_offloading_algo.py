import pytest
from pytest import mark

from src.offloading_algo.offloading_algo import OffloadingAlgo


@mark.parametrize(
    "latency, synthetic_latency, expected_offloading_layer_index",
    [
        (
                0.000000001,
                0.000000001,
                2
        ),
        (
                float('inf'),
                float('inf'),
                4,
        ),
    ],
)
def test_offloading_algo(
        latency, synthetic_latency, expected_offloading_layer_index,
        layers_sizes_offloading_data, edge_offloading_data, device_offloading_data):
    test_latency = latency * synthetic_latency
    avg_speed = max(layers_sizes_offloading_data) / test_latency

    offloading_algo = OffloadingAlgo(
        avg_speed=avg_speed,
        num_layers=len(layers_sizes_offloading_data) - 1,
        layers_sizes=list(layers_sizes_offloading_data),
        inference_time_device=list(device_offloading_data),
        inference_time_edge=list(device_offloading_data)
    )
    best_offloading_layer = offloading_algo.static_offloading()
    assert best_offloading_layer <= expected_offloading_layer_index


if __name__ == "__main__":
    pytest.main()
