from streamlit import file_uploader, image, set_page_config, tabs

from utils import binarise, display_text, grayscale, open_image

set_page_config(
    page_title="Digitise Form", page_icon="📝", initial_sidebar_state="expanded"
)
image_display_size = 300
uploaded_file = file_uploader("Choose an image")
original_tab, grayscale_tab, binary_tab = tabs(["Original", "Grayscale", "Binary"])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image_data = open_image(image_bytes=bytes_data)
    gray_data = grayscale(image_data=image_data)
    binary_data = binarise(image_data=gray_data)
    with original_tab:
        image(bytes_data, width=image_display_size)
        display_text(image_data=image_data, name="original")
    with grayscale_tab:
        image(gray_data, width=image_display_size)
        display_text(image_data=gray_data, name="grayscale")
    with binary_tab:
        image(binary_data, width=image_display_size)
        display_text(image_data=binary_data, name="binary")
