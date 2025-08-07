from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import JSONResponse
from API.sama_updated import read_docx, process_pdf, convert_to_markdown, extract_text_from_pptx
from API.florence import process_file_with_florence
from pydantic import BaseModel, Field
from API.googlevision import process_file_with_google_vision
from API.caludeocr import process_file_with_claude
from API.getllm import get_llm
from API.open import process_file_with_gpt_vision
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
import os

app = FastAPI()

# Directory to store uploaded files
uploads = "uploads"
os.makedirs(uploads, exist_ok=True)

# Define KO Class
class KO(BaseModel):
    Query: str = Field(..., description='the root cause of the problem')
    Symptoms: str = Field(..., description='Symptoms refer to the observable signs, behaviors, or effects that indicate the presence of a problem/issue')
    Short_description: str = Field(..., description='A brief and short description of user query')
    Long_description: str = Field(..., description='A detailed and good enhanced description of user query')
    Causes: str = Field(..., description='Causes refer to the underlying reasons or factors that lead to the occurrence of a problem or issue')
    Resolution_note: str = Field(..., description='step by step detailed Enhanced knowledge article which covers all the scenarios')
    Relevancy: str = Field(..., description="Relevancy of the Knowledge article, between [0-100]%")


# Route for file upload and KO generation
@app.post("/upload-file/")
async def upload_file(
    file: UploadFile = File(...), 
    model_name: str = Form("azureai"), 
    ocr_method: str = Form("tesseract")
):
    # Save the uploaded file
    uploads_folder = "uploads"
    os.makedirs(uploads_folder, exist_ok=True)
    file_path = os.path.join(uploads_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if ocr_method == "tesseract":
        # Process the file based on type
        if file.filename.endswith(".pdf"):
            content = process_pdf(file_path)
        elif file.filename.endswith(".docx"):
            content = read_docx(file_path)
        elif file.filename.endswith(".pptx"):
            content = extract_text_from_pptx(file_path)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported file format."})
        all_text = ""
        for item in content:
            all_text += convert_to_markdown(item)

    # Perform OCR
    if ocr_method == "openai":
        all_text = process_file_with_gpt_vision(file_path, uploads_folder, verbose=True)

    elif ocr_method == "florence":
        all_text = process_file_with_florence(file_path, uploads_folder, verbose=True)
        all_text = all_text[0].get("<OCR>")

    elif ocr_method == "google":
        all_text = process_file_with_google_vision(file_path, uploads_folder, verbose=True)

    elif ocr_method == "claude":
        all_text = process_file_with_claude(file_path, uploads_folder, verbose=True)
        all_text = all_text[0].get("text")

    # Get LLM based on model name
    llm = get_llm(model_name)

    # Generate the KO using the selected LLM
    ko_object = await generate_ko(all_text, llm)

    return {"filename": file.filename, "KO_Article": ko_object}

async def generate_ko(mktxt: str, llm):
    # Create prompt templates
    template1 = """You are an IT assistant for Knowledge article generation using data extracted from documents. You MUST provide the following fields for the user's query:
                   Short description, Long description (detailed and enhanced), Causes (underlying reasons for the problem),
                   Resolution_note (step-by-step enhanced knowledge article covering all scenarios), and Symptoms (observable signs indicating the problem).
                   Provide the response **strictly in English**."""
                   
    template2 = """\nDocument information: {mktxt}\n\nStrictly follow the specified format: {format_instructions}.
                   Provide all the details, and make sure the language used is English only."""

    # Combine the templates into one prompt
    template = template1 + template2 + "\n\nProvide the response in English only, and ensure each field is filled. Here is the KO object:"

    # Set up the prompt parser and retry mechanism
    parser = PydanticOutputParser(pydantic_object=KO)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=5)

    # Create the final prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["mktxt"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Define completion chain
    completion_chain = prompt | llm

    if llm.get_name() == "Ollama":
        main_chain = RunnableParallel(
            completion=completion_chain,
            prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(x['completion'], x['prompt_value']))

    else:
        # Run everything in parallel with retry logic
        main_chain = RunnableParallel(
            completion=completion_chain,
            prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(x['completion'].content, x['prompt_value']))

    # Run the chain and get the parsed output
    ko_object = main_chain.invoke(mktxt)

    return ko_object

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
