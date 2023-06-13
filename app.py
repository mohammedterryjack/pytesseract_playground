from pytesseract import get_tesseract_version
from streamlit import file_uploader, image, set_page_config, sidebar, slider, tabs

from utils import (
    binarise,
    display_text,
    draw_bounding_boxes,
    grayscale,
    invert,
    open_image,
)

set_page_config(
    page_title=f"Tesseract {get_tesseract_version()}",
    page_icon="📝",
)
image_display_size = 300
uploaded_file = file_uploader("Choose an image")
distance = sidebar.slider("Min Distance from Field", 0, 100, 80)
show_boxes = sidebar.checkbox("Show Fields")
original_tab, grayscale_tab, binary_tab = tabs(["Original", "Grayscale", "Binary"])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image_data = open_image(image_bytes=bytes_data)
    gray_data = grayscale(image_data=image_data)
    binary_data = binarise(image_data=gray_data)
    binary_data_inverted = invert(binary_data)
    with original_tab:
        bytes_data_bboxes, coordinates_fields, coordinates_boxes = draw_bounding_boxes(
            image=image_data, binarised_image=binary_data
        )
        image(
            bytes_data_bboxes if show_boxes else bytes_data,
            width=image_display_size,
            channels="BGR" if show_boxes else "RGB",
        )
        display_text(
            image_data=image_data,
            name="original",
            coordinates=coordinates_fields + coordinates_boxes,
            min_field_distance=distance,
        )
    with grayscale_tab:
        gray_data_bboxes, coordinates_fields, coordinates_boxes = draw_bounding_boxes(
            image=gray_data, binarised_image=binary_data
        )
        image(gray_data_bboxes if show_boxes else gray_data, width=image_display_size)
        display_text(
            image_data=gray_data,
            name="grayscale",
            coordinates=coordinates_fields + coordinates_boxes,
            min_field_distance=distance,
        )
    with binary_tab:
        (
            binary_data_inverted_bboxes,
            coordinates_fields,
            coordinates_boxes,
        ) = draw_bounding_boxes(image=binary_data_inverted, binarised_image=binary_data)
        image(
            binary_data_inverted_bboxes if show_boxes else binary_data_inverted,
            width=image_display_size,
        )
        display_text(
            image_data=binary_data,
            name="binary",
            coordinates=coordinates_fields + coordinates_boxes,
            min_field_distance=distance,
        )
