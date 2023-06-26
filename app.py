from pytesseract import get_tesseract_version
from streamlit import file_uploader, image, set_page_config, sidebar, tabs

from image_pipeline import ImagePipeline
from utils import display_text, open_image

set_page_config(
    page_title=f"Tesseract {get_tesseract_version()}",
    page_icon="📝",
)
image_display_size = 300
uploaded_file = file_uploader("Choose an image")
min_area = sidebar.slider("Minimum Area", 0, 100, 30)
k_colours = sidebar.slider("Colour Buckets", 2, 5, 2)

image_pipeline = ImagePipeline(n_colours=k_colours, min_area=min_area)

original_tab, clustered_tab, gray_tab, binary_tab, box_tab = tabs(
    ["Colour", "Clustered", "Grayscale", "Black and White", "Bounding Boxes"]
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
    ) = image_pipeline(colour_image=image_data)

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
    display_text(
        image_data=image_data,
        name="Text",
        coordinates=[],
        min_field_distance=0,
    )
