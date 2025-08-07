import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import os
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import copy
from st_copy_to_clipboard import st_copy_to_clipboard

# Set up device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Cache the model and processor to avoid reloading
@st.cache_resource
def load_model_and_processor():
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        return imports

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    return model, processor

# Function to process image through the model for OCR, Caption, or Object Detection tasks
def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    model, processor = load_model_and_processor()

    if task_prompt == '<OCR>':
        image = image.convert("RGB")

    image = np.array(image)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cpu', torch.float32)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cpu(),
        pixel_values=inputs["pixel_values"].cpu(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.shape[1], image.shape[0])
    )

    return parsed_answer

# Common function to draw polygons for Referring Expression Segmentation
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue

            _polygon = (_polygon).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    return image

# Function to plot bounding boxes for Object Detection
def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    st.pyplot(fig)

# Function to convert detection results to object detection format
def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    
    return od_results

# Function to handle the OCR task and display the result with a "copy to clipboard" button
def display_ocr_result(ocr_text):
    st.write("OCR Result:")
    # Create a text area where the OCR result can be displayed with scroll
    st.text_area("OCR Output", ocr_text,st_copy_to_clipboard(ocr_text),height=400,key='ocr_output')
