from pytesseract import get_tesseract_version
from streamlit import (
    file_uploader,
    form,
    form_submit_button,
    image,
    set_page_config,
    sidebar,
    slider,
    tabs,
    write,
)
from streamlit_tags import st_tags

from image_pipeline import ImagePipeline
from text_extraction_pipeline import TextExtractionPipeline
from utils import open_image

set_page_config(
    page_title=f"Tesseract {get_tesseract_version()}",
    page_icon="📝",
)
image_display_size = 300
uploaded_file = file_uploader("Choose an image")
min_area = sidebar.slider("Minimum Area", 0, 100, 30)
k_colours = sidebar.slider("Colour Buckets", 2, 5, 2)
character_pixels = sidebar.slider("Character Pixel Resolution", 30, 50, 40)
field_distance = sidebar.slider("Radius from field", 0, 300, 200)

image_pipeline = ImagePipeline(n_colours=k_colours, min_area=min_area)

original_tab, clustered_tab, gray_tab, binary_tab, box_tab, preprocess_tab = tabs(
    [
        "Colour",
        "Clustered",
        "Grayscale",
        "Black and White",
        "Bounding Boxes",
        "Text Preprocessing",
    ]
)
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image_data = open_image(image_bytes=bytes_data)
    (
        image_colour,
        image_clustered,
        image_grey,
        image_binary,
        image_bbox,
        bboxes,
    ) = image_pipeline(colour_image=image_data)
    text_pipeline = TextExtractionPipeline(
        optimal_character_height_in_pixels=character_pixels, field_boxes=bboxes
    )
    image_data_preprocessed = text_pipeline.preprocess_image(image_data=image_data)

    with original_tab:
        image(
            bytes_data,
            width=image_display_size,
            channels="RGB",
        )
    with clustered_tab:
        image(
            image_clustered,
            width=image_display_size,
            channels="BGR",
        )
    with gray_tab:
        image(
            image_grey,
            width=image_display_size,
            channels="RGB",
        )
    with binary_tab:
        image(
            image_binary,
            width=image_display_size,
            channels="RGB",
        )
    with box_tab:
        image(
            image_bbox,
            width=image_display_size,
            channels="BGR",
        )
    with preprocess_tab:
        image(image_data_preprocessed)
    confidence = slider("Minimum Confidence", 0, 100)
    with form(key="OCR"):
        if form_submit_button("Extract Text"):
            texts = text_pipeline.extract_text_data(
                image_data=image_data_preprocessed, minimum_confidence=confidence
            )
            fields = list(
                text_pipeline.extract_field_labels(
                    text_data=texts, min_distance=field_distance
                )
            )
            field_names = st_tags(
                label=":red[Extracted Fields]",
                value=list(map(lambda field: field["text"], fields)),
            )
            write(field_names)
            write(":blue[Complete Text:]")
            for text in texts:
                write(text["text"])
