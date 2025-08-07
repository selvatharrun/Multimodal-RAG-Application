import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import copy
from pages.imports.florencevlm import load_model_and_processor, run_example, plot_bbox, draw_polygons,convert_to_od_format,display_ocr_result
# Set up the layout
st.markdown(
    """
    <style>
    .main {
        padding-left: 0rem;
        padding-right: 0rem;
        background:black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

load_model_and_processor()
# Streamlit app
st.title(":rainbow[**Image Task Runner with Florence-2 Model**]")

# Create columns: left_col, mid_col, right_col
left_col, right_col = st.columns([2, 2])

with left_col:
    st.header("Image Upload and Introduction")
    st.write("Upload an image and select a task to run.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    st.header("Choose Task")
    task_selected = st.selectbox("Select a task:", ["", "Caption", "Object Detection", "Dense Region Caption", "Region Proposal", "OCR", "Caption to Phrase Grounding", "Referring Expression Segmentation", "Region to Segmentation"])

with right_col:
    if uploaded_file is not None and task_selected == "Open Vocabulary Detection":
        text_input = st.text_input("Enter open vocabulary query (e.g., 'a green car'):", "a green car")
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"

        st.write("Running Open Vocabulary Detection...")
        results = run_example(image, task_prompt, text_input)
        
        # Convert results to object detection format
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])

        # Plot the bounding boxes on the image
        plot_bbox(image, bbox_results)

    if uploaded_file is not None and task_selected == "Referring Expression Segmentation":
        # Text input for referring expression
        text_input = st.text_input("Enter referring expression (e.g., 'a green car'):", "a green car")
        task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"

        st.write("Running Referring Expression Segmentation...")
        results = run_example(image, task_prompt, text_input)
        
        # Create a deep copy of the image for drawing polygons
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results["<REFERRING_EXPRESSION_SEGMENTATION>"], fill_mask=True)

        # Display the result image
        st.image(output_image, caption="Referring Expression Segmentation Result")

    if task_selected == "Caption":
        # Show options for caption generation
        caption_option = st.radio(
            "Choose Caption Type:",
            ['Basic Caption', 'Detailed Caption', 'More Detailed Caption'],
            index=0
        )
        # Map the caption option to task prompts
        task_prompt_map = {
            'Basic Caption': '<CAPTION>',
            'Detailed Caption': '<DETAILED_CAPTION>',
            'More Detailed Caption': '<MORE_DETAILED_CAPTION>'
        }
        task_prompt = task_prompt_map[caption_option]

        st.write(f"Generating {caption_option}...")
        result = run_example(image, task_prompt)
        st.write(f"Caption: {result[task_prompt]}")

    if task_selected == "Object Detection":
        # Perform object detection
        st.write("Running Object Detection...")
        task_prompt = '<OD>'
        results = run_example(image, task_prompt)
        plot_bbox(image, results["<OD>"])

    if task_selected == "Dense Region Caption":
        task_prompt = "<DENSE_REGION_CAPTION>"
        results = run_example(image, task_prompt)
        st.write("Running Dense Region Captioning...")
        plot_bbox(image, results['<DENSE_REGION_CAPTION>'])

    if task_selected == "Region Proposal":
        task_prompt = "<REGION_PROPOSAL>"
        results = run_example(image, task_prompt)
        st.write("Running Region Proposal...")
        plot_bbox(image, results['<REGION_PROPOSAL>'])

    if task_selected == "OCR":            
        task_prompt = '<OCR>'
        st.write("Running OCR...")
        result = run_example(image, task_prompt)
        display_ocr_result(result[task_prompt])
    
    if task_selected == "Caption to Phrase Grounding":
        text_input = st.text_input("Enter additional text input (optional)", "")         
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        st.write("Running Caption to Phrase Grounding...")
        results = run_example(image, task_prompt, text_input)
        plot_bbox(image, results[task_prompt])
    
    # Example for Referring Expression Segmentation
    if uploaded_file is not None and task_selected == "Referring Expression Segmentation":
        text_input = st.text_input("Enter referring expression (e.g., 'a green car'):", "a green car")
        task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"

        st.write("Running Referring Expression Segmentation...")
        results = run_example(image, task_prompt, text_input)
        
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results["<REFERRING_EXPRESSION_SEGMENTATION>"], fill_mask=True)

        st.image(output_image, caption="Referring Expression Segmentation Result")

