from streamlit import (
    error,
    file_uploader,
    form,
    form_submit_button,
    image,
    set_page_config,
    slider,
    write,
)

from utils import extract_text

set_page_config(
    page_title="Digitise Form", page_icon="📝", initial_sidebar_state="expanded"
)

uploaded_file = file_uploader("Choose an image")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image(bytes_data, width=300)

with form(key="ocr"):
    confidence = slider("Minimum Confidence", 0, 100)
    if form_submit_button("Extract Text"):
        if uploaded_file is None:
            error("Please upload an image")
        else:
            texts = extract_text(image_bytes=bytes_data, confidence=confidence)
            for text in texts:
                for section in text.split("\t"):
                    write(section)
