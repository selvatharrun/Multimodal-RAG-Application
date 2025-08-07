# This code capturing table content and tables ,colors,font size and convert into md files
import os
import io
from docx import Document
from docx.shape import InlineShape
from docx.oxml.ns import qn
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from docx.shape import InlineShape
from PIL import Image
import pytesseract
import xml.etree.ElementTree as ET

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def read_docx(file_path):
    doc = Document(file_path)
    full_content = []
    
    # Define namespaces
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    }

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            paragraph_content = {
                'type': 'paragraph',
                'text': block.text,
                'style': block.style.name,
                'bold': False,
                'italic': False,
                'underline': False,
                'font_size': None,
                'font_color': None
            }
            
            # Check for text formatting
            for run in block.runs:
                if run.bold:
                    paragraph_content['bold'] = True
                if run.italic:
                    paragraph_content['italic'] = True
                if run.underline:
                    paragraph_content['underline'] = True
                if run.font.size:
                    paragraph_content['font_size'] = run.font.size
                if run.font.color.rgb:
                    paragraph_content['font_color'] = run.font.color.rgb
            
            full_content.append(paragraph_content)
            
            # Process inline shapes (which include images)
            for run in block.runs:
                drawing_elements = run._element.findall('.//w:drawing', namespaces=namespaces)
                for drawing in drawing_elements:
                    inline = drawing.find('.//wp:inline', namespaces=namespaces)
                    if inline is not None:
                        blip = inline.find('.//a:blip', namespaces=namespaces)
                        if blip is not None:
                            image_rid = blip.get(qn('r:embed'))
                            image_part = doc.part.related_parts[image_rid]
                            image_stream = io.BytesIO(image_part.blob)
                            image = Image.open(image_stream)
                            
                            # Perform OCR on the image
                            text = pytesseract.image_to_string(image)
                            
                            extent = inline.find('.//wp:extent', namespaces=namespaces)
                            width = int(extent.get('cx')) if extent is not None else 0
                            height = int(extent.get('cy')) if extent is not None else 0
                            
                            full_content.append({
                                'type': 'image',
                                'text': text,
                                'width': width,
                                'height': height
                            })
        
        elif isinstance(block, Table):
            table_data = []
            for row in block.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            full_content.append({
                'type': 'table',
                'data': table_data
            })
    
    return full_content

def convert_to_markdown(content):
    if content['type'] == 'paragraph':
        text = content['text']
        style = content['style']
        
        if style.startswith('Heading'):
            try:
                level = int(''.join(filter(str.isdigit, style)))
            except ValueError:
                level = 1
            return f"{'#' * level} {text}\n\n"
        
        formatting = []
        if content['bold']:
            formatting.append('**')
        if content['italic']:
            formatting.append('*')
        if content['underline']:
            formatting.append('__')
        
        formatted_text = f"{''.join(formatting)}{text}{''.join(reversed(formatting))}"
        
        color_info = f"[Color: {content['font_color']}]" if content['font_color'] else ""
        size_info = f"[Size: {content['font_size']}]" if content['font_size'] else ""
        
        return f"{formatted_text} {color_info}{size_info}\n\n"
    
    elif content['type'] == 'table':
        table_md = ""
        for i, row in enumerate(content['data']):
            table_md += "| " + " | ".join(row) + " |\n"
            if i == 0:
                table_md += "|" + "|".join(["---" for _ in row]) + "|\n"
        return table_md + "\n"
    
    elif content['type'] == 'image':
        return f"[Image Content: {content['text']}]\n\n"

# Main execution
if __name__ == "__main__":
    # Specify the paths to your .docx files
    file_paths = ["testddoc.docx"]

    for file_path in file_paths:
        print(f"Processing {file_path}")
        
        # Step 1: Read the document
        content = read_docx(file_path)

        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create the output filename
        output_file = f"{base_filename}_converted.md"

        # Convert to markdown and write to a file
        with open(output_file, "w", encoding="utf-8") as f:
            for item in content:
                f.write(convert_to_markdown(item))

        print(f"Converted content has been written to '{output_file}'\n")

    print("All files have been processed and converted to markdown.")