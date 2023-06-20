from io import BytesIO
from typing import Dict, Generator, List, Tuple

from cv2 import (
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY,
    MORPH_DILATE,
    MORPH_RECT,
    RETR_TREE,
    THRESH_BINARY_INV,
    THRESH_OTSU,
    bitwise_not,
    boundingRect,
    cvtColor,
    findContours,
    getStructuringElement,
    morphologyEx,
    rectangle,
    threshold,
)
from numpy import array, ndarray
from PIL import Image
from pytesseract import image_to_data
from streamlit import cache_data, form, form_submit_button, slider, write


def display_text(
    image_data: ndarray,
    name: str,
    coordinates: List[Tuple[int, int]],
    min_field_distance: int,
) -> None:
    with form(key=name):
        confidence = slider("Minimum Confidence", 0, 100)
        if form_submit_button("Extract Text"):
            results = extract_text_from_image(image_data=image_data)
            confident_results = list(
                filter_results_below_confidence(results=results, confidence=confidence)
            )
            confident_results = annotate_field_text(
                results=confident_results,
                coordinates_of_fields=coordinates,
                min_distance_threshold=min_field_distance,
            )
            grouped_text = group_text_by_block_number(results=confident_results)
            for text in grouped_text:
                write(text)


def binarise(image_data: ndarray) -> ndarray:
    _, image_threshold = threshold(image_data, 0, 255, THRESH_BINARY_INV | THRESH_OTSU)
    return image_threshold


def invert(image_data: ndarray) -> ndarray:
    return bitwise_not(image_data)


def grayscale(image_data: ndarray) -> ndarray:
    return cvtColor(image_data, COLOR_BGR2GRAY)


def open_image(image_bytes: str) -> ndarray:
    image_data = BytesIO(image_bytes)
    image_object = Image.open(image_data)
    return array(image_object)


@cache_data
def extract_text_from_image(image_data: ndarray) -> Dict[str, str]:
    result = image_to_data(image_data)
    return list(format_ocr_result_as_dictionary(ocr_result=result))


def filter_results_below_confidence(
    results: List[Dict[str, str]], confidence: float
) -> Generator[Dict[str, str], None, None]:
    for result in results:
        if "conf" not in result:
            continue
        if float(result["conf"]) >= confidence:
            yield result


def group_text_by_block_number(
    results: List[Dict[str, str]], delimiter: str = " "
) -> List[str]:
    sections = {}
    for result in results:
        if "text" not in result:
            continue
        text = result["text"]
        if not text:
            continue
        if result["field"]:
            text = f":red[{text}]"
        n = result["block_num"]
        if n in sections:
            sections[n] += delimiter + text
        else:
            sections[n] = text
    return list(sections.values())


def format_ocr_result_as_dictionary(
    ocr_result: str, line_delimiter: str = "\n", key_delimiter: str = "\t"
) -> Generator[Dict[str, str], None, None]:
    lines = ocr_result.split(line_delimiter)
    header = lines.pop(0)
    keys = header.split(key_delimiter)
    for line in lines:
        values = line.split(key_delimiter)
        key_value_pairs = zip(keys, values)
        yield dict(key_value_pairs)


def blur_image(image_data: ndarray) -> ndarray:
    return morphologyEx(
        image_data, MORPH_DILATE, getStructuringElement(MORPH_RECT, (3, 3))
    )


def annotate_field_text(
    results: List[Dict[str, str]],
    coordinates_of_fields: List[Tuple[int, int]],
    min_distance_threshold: int,
) -> List[Dict[str, str]]:
    for result in results:
        if (
            "left" not in result
            or "top" not in result
            or not any(coordinates_of_fields)
        ):
            result["field"] = False
            continue
        x = int(result["left"])
        y = int(result["top"])
        nearest_coordinates = min(
            coordinates_of_fields,
            key=lambda coordinate: l2_distance(
                coordinates1=coordinate, coordinates2=(x, y)
            ),
        )
        nearest_distance = l2_distance(
            coordinates1=nearest_coordinates, coordinates2=(x, y)
        )
        result["field"] = nearest_distance < min_distance_threshold
    return results


def l2_distance(coordinates1: Tuple[int, int], coordinates2: Tuple[int, int]) -> float:
    x1, y1 = coordinates1
    x2, y2 = coordinates2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@cache_data
def draw_bounding_boxes(
    image: ndarray,
    binarised_image: ndarray,
    noise_threshold: int = 100,
    checkbox_threshold: int = 10,
    rectangle_ratio: int = 10,
) -> Tuple[ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    bbox_image = image.copy()
    image_blurred = blur_image(image_data=binarised_image)
    contours, _ = findContours(image_blurred, RETR_TREE, CHAIN_APPROX_SIMPLE)

    field_coordinates, checkbox_coordinates = [], []

    for contour in contours:
        x, y, w, h = boundingRect(contour)
        if w * h < noise_threshold:
            continue
        if abs(w - h) <= checkbox_threshold:
            rectangle(bbox_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            checkbox_coordinates.append((x, y))
        elif w > h and w / h < rectangle_ratio:
            rectangle(bbox_image, (x, y), (x + w, y + h), (255, 0, 255), 3)
            field_coordinates.append((x, y))
    return bbox_image, field_coordinates, checkbox_coordinates
