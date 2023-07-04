from typing import Dict, Generator, List, Tuple

from cv2 import (
    CC_STAT_AREA,
    CC_STAT_HEIGHT,
    CC_STAT_LEFT,
    CC_STAT_TOP,
    CC_STAT_WIDTH,
    COLOR_BGR2GRAY,
    COLOR_RGBA2RGB,
    CV_32S,
    THRESH_BINARY_INV,
    THRESH_OTSU,
    connectedComponentsWithStats,
    cvtColor,
    rectangle,
    threshold,
)
from numpy import ndarray
from sklearn.cluster import KMeans


class ImagePipeline:
    """image pipeline to extract form fields using colour clustering and connected-component analysis"""

    def __init__(self, n_colours: int, min_area: int) -> None:
        self.colour_clusterer = KMeans(n_clusters=n_colours)
        self.PIXEL_OFF = 0
        self.PIXEL_ON = 255
        self.min_area = min_area

    def __call__(
        self, colour_image: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, List[Dict[str, int]]]:
        image_colour = self.rgb_image(colour_image=colour_image)
        image_simplified = self.repaint_image(colour_image=image_colour)
        image_gray = self.grayscale_image(colour_image=image_simplified)
        image_binary = self.binary_image(
            grayscale_image=image_gray,
            min_value=self.PIXEL_OFF,
            max_value=self.PIXEL_ON,
        )
        bboxes = list(
            self.bounding_boxes(binary_image=image_binary, min_area=self.min_area)
        )
        image_annotated = self.annotate_boxes(colour_image=colour_image, boxes=bboxes)
        return (
            image_colour,
            image_simplified,
            image_gray,
            image_binary,
            image_annotated,
            bboxes,
        )

    def repaint_image(self, colour_image: ndarray) -> ndarray:
        width, height, colour_channels = colour_image.shape
        n_pixels = width * height
        pixels = colour_image.copy().reshape((n_pixels, colour_channels))
        self.colour_clusterer.fit(pixels)
        repainted_pixels = self.colour_clusterer.predict(pixels)
        image_repainted_cluster_ids = repainted_pixels.reshape((width, height))
        image_repainted = self.colour_clusterer.cluster_centers_[
            image_repainted_cluster_ids
        ]
        return image_repainted.astype("uint8") * self.PIXEL_ON

    @staticmethod
    def rgb_image(colour_image: ndarray) -> ndarray:
        return cvtColor(colour_image, COLOR_RGBA2RGB)

    @staticmethod
    def grayscale_image(colour_image: ndarray) -> ndarray:
        return cvtColor(colour_image, COLOR_BGR2GRAY)

    @staticmethod
    def binary_image(
        grayscale_image: ndarray, min_value: int, max_value: int
    ) -> ndarray:
        _, image_binary = threshold(
            grayscale_image, min_value, max_value, THRESH_BINARY_INV | THRESH_OTSU
        )
        return image_binary

    @staticmethod
    def bounding_boxes(
        binary_image: ndarray, min_area: int
    ) -> Generator[Dict[str, int], None, None]:
        _, _, statistics, _ = connectedComponentsWithStats(binary_image, 4, CV_32S)
        for i, stats in enumerate(statistics):
            if i and stats[CC_STAT_AREA] > min_area:
                yield {
                    "left": stats[CC_STAT_LEFT],
                    "top": stats[CC_STAT_TOP],
                    "width": stats[CC_STAT_WIDTH],
                    "height": stats[CC_STAT_HEIGHT],
                    "right": stats[CC_STAT_LEFT] + stats[CC_STAT_WIDTH],
                    "bottom": stats[CC_STAT_TOP] + stats[CC_STAT_HEIGHT],
                }

    @staticmethod
    def annotate_boxes(
        colour_image: ndarray,
        boxes: List[Dict[str, int]],
        box_colour: Tuple[int, int, int] = (255, 0, 255),
    ) -> ndarray:
        annotated_image = colour_image.copy()
        for box in boxes:
            rectangle(
                annotated_image,
                (box["left"], box["top"]),
                (box["right"], box["bottom"]),
                box_colour,
                3,
            )
        return annotated_image
