from io import BytesIO

from numpy import array, ndarray
from PIL import Image


def open_image_from_file(filename: str) -> ndarray:
    image_object = Image.open(filename)
    return array(image_object)


def open_image(image_bytes: str) -> ndarray:
    image_data = BytesIO(image_bytes)
    return open_image_from_file(filename=image_data)
