from io import BytesIO

from numpy import array, ndarray
from PIL import Image


def open_image(image_bytes: str) -> ndarray:
    image_data = BytesIO(image_bytes)
    image_object = Image.open(image_data)
    return array(image_object)
