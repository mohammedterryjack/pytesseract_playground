from io import BytesIO
from typing import Dict, Generator, List

from cv2 import (
    COLOR_BGR2GRAY,
    THRESH_BINARY_INV,
    THRESH_OTSU,
    bitwise_not,
    cvtColor,
    threshold,
)
from numpy import array, ndarray
from PIL import Image
from pytesseract import image_to_data
from streamlit import cache_data, form, form_submit_button, slider, write


def display_text(image_data: ndarray, name: str) -> None:
    with form(key=name):
        confidence = slider("Minimum Confidence", 0, 100)
        if form_submit_button("Extract Text"):
            results = extract_text_from_image(image_data=image_data)
            confident_results = filter_results_below_confidence(
                results=results, confidence=confidence
            )
            grouped_text = group_text_by_block_number(results=confident_results)
            for text in grouped_text:
                for section in text.split("\t"):
                    write(section)


def binarise(image_data: ndarray) -> ndarray:
    _, image_threshold = threshold(image_data, 0, 255, THRESH_BINARY_INV | THRESH_OTSU)
    return bitwise_not(image_threshold)


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


def group_text_by_block_number(results: List[Dict[str, str]]) -> List[str]:
    sections = {}
    pointer = {}
    for result in results:
        if "text" not in result:
            continue
        text = result["text"]
        if not text:
            continue
        n = result["block_num"]
        position = int(result["left"]) + int(result["width"])
        delimiter = "\t" if position - pointer.get(n, 0) >= 100 else " "
        if n in sections:
            sections[n] += delimiter + text
        else:
            sections[n] = text
        pointer[n] = position
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
