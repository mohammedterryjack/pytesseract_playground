from argparse import ArgumentParser

from image_pipeline import ImagePipeline
from text_extraction_pipeline import TextExtractionPipeline
from utils import open_image_from_file

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, default="images/segmented/section1.png")
    parser.add_argument("--confidence", type=int, default=0)
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--k_colours", type=int, default=2)
    parser.add_argument("--character_pixels", type=int, default=40)
    parser.add_argument("--field_distance", type=int, default=200)
    arguments = parser.parse_args()

    image_pipeline = ImagePipeline(
        n_colours=arguments.k_colours, min_area=arguments.min_area
    )
    image_data = open_image_from_file(filename=arguments.filename)
    _, _, _, _, _, bboxes = image_pipeline(colour_image=image_data)

    text_pipeline = TextExtractionPipeline(
        optimal_character_height_in_pixels=arguments.character_pixels,
        field_boxes=bboxes,
    )
    image_data_preprocessed = text_pipeline.preprocess_image(image_data=image_data)
    texts = text_pipeline.extract_text_data(
        image_data=image_data_preprocessed, minimum_confidence=arguments.confidence
    )
    fields = list(
        text_pipeline.extract_field_labels(
            text_data=texts, min_distance=arguments.field_distance
        )
    )
    results = list(
        map(
            lambda field: {
                "extracted_block_type": "field",
                "value": field["text"],
                "data_type": "string",
                "cluster": None,
            },
            fields,
        )
    )
    print(results)
