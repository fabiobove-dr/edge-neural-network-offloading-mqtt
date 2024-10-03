from dataclasses import dataclass


@dataclass
class ModelManagerConfig:
    DEFAULT_MODEL_NAME: str = "resnet_model.keras"
    MODEL_DIR_PATH: str = "resnet_model"
    MODEL_PATH: str = f"{MODEL_DIR_PATH}/{DEFAULT_MODEL_NAME}"
    IMAGE_SIZE: int = 10
    SAVE_PATH: str = f"../"
