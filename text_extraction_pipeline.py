from statistics import mode
from typing import Dict, Generator, List, Tuple, Union

from cv2 import COLOR_BGR2GRAY, cvtColor, filter2D, resize
from numpy import array, ndarray
from pytesseract import Output, image_to_data


class TextExtractionPipeline:
    def __init__(
        self,
        field_boxes: List[Dict[str, int]],
        optimal_character_height_in_pixels: int = 36,
    ) -> None:
        self._optimal_character_height_pixels = optimal_character_height_in_pixels
        self.bboxes = field_boxes
        self.ratio = self.aspect_ratio(
            character_height=mode(map(lambda bbox: bbox["height"], self.bboxes)),
            optimal_character_height=self._optimal_character_height_pixels,
        )

    def preprocess_image(self, image_data: ndarray) -> ndarray:
        image_data = self.sharpen_image(image_data=image_data)
        image_data = self.grayscale_image(colour_image=image_data)
        image_data = self.resize_image(
            image=image_data,
            ratio=self.ratio,
        )
        return image_data

    def extract_field_labels(
        self,
        text_data: List[Dict[str, Union[str, int]]],
        min_distance: int,
    ) -> Generator[Dict[str, Union[str, int]], None, None]:
        taken_boxes = set()
        for text in text_data:
            if not any(text["text"].strip()):
                continue
            coordinates_text = (text["right"], text["bottom"])
            nearest_distance = None
            nearest_box_index = None
            for i, box in enumerate(self.bboxes):
                if i in taken_boxes:
                    continue
                coordinates_field = (box["left"] * self.ratio, box["top"] * self.ratio)
                distance = TextExtractionPipeline.l2_distance(
                    coordinates1=coordinates_field, coordinates2=coordinates_text
                )
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_box_index = i
            if nearest_distance <= min_distance:
                yield text
                taken_boxes.add(nearest_box_index)

    @staticmethod
    def resize_image(image: ndarray, ratio: float) -> ndarray:
        width, height = image.shape
        width *= ratio
        height *= ratio
        return resize(image, (int(height), int(width)))

    @staticmethod
    def aspect_ratio(character_height: int, optimal_character_height: int) -> float:
        return optimal_character_height / character_height

    @staticmethod
    def grayscale_image(colour_image: ndarray) -> ndarray:
        return cvtColor(colour_image, COLOR_BGR2GRAY)

    @staticmethod
    def sharpen_image(image_data: ndarray) -> ndarray:
        kernel = array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return filter2D(image_data, -1, kernel)

    @staticmethod
    def extract_text_data(image_data: ndarray, minimum_confidence: int) -> List[str]:
        result = image_to_data(image_data, output_type=Output.DICT)
        confident_results = TextExtractionPipeline.filter_results_below_confidence(
            results=result, min_confidence=minimum_confidence
        )
        return TextExtractionPipeline.group_text_by_block_number(
            results=confident_results
        )

    @staticmethod
    def filter_results_below_confidence(
        results: List[Dict[str, str]], min_confidence: float
    ) -> Dict[str, List[str]]:
        removed_indexes = []
        for index, confidence in enumerate(results["conf"]):
            if confidence < min_confidence:
                removed_indexes.append(index)
        for key, values in results.items():
            results[key] = list(
                value for i, value in enumerate(values) if i not in removed_indexes
            )
        return results

    @staticmethod
    def group_text_by_block_number(
        results: Dict[str, List[str]], delimiter: str = " "
    ) -> List[Dict[str, Union[str, int]]]:
        sections = {}
        for index in range(len(results["text"])):
            block_number = results["block_num"][index]
            text = results["text"][index]
            left = results["left"][index]
            top = results["top"][index]
            if block_number in sections:
                sections[block_number]["text"] += delimiter + text
                sections[block_number]["left"] = min(
                    sections[block_number]["left"], left
                )
                sections[block_number]["top"] = max(sections[block_number]["top"], top)
                sections[block_number]["right"] = max(
                    sections[block_number]["right"], left
                )
                sections[block_number]["bottom"] = min(
                    sections[block_number]["top"], top
                )
            else:
                sections[block_number] = {
                    "text": text,
                    "left": left,
                    "top": top,
                    "right": left,
                    "bottom": top,
                }
        return list(sections.values())

    @staticmethod
    def l2_distance(
        coordinates1: Tuple[int, int], coordinates2: Tuple[int, int]
    ) -> float:
        x1, y1 = coordinates1
        x2, y2 = coordinates2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
