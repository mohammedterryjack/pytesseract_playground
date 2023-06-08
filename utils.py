from io import BytesIO
from typing import Dict, Generator, List

from PIL import Image
from pytesseract import image_to_data
from streamlit import cache_data


@cache_data
def extract_text_from_image(image_bytes: str) -> Dict[str, str]:
    image_object = Image.open(BytesIO(image_bytes))
    result = image_to_data(image_object)
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
