from langchain_core.runnables import RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
import asyncio
import API.open as open
import gc, os, json
import base64
import configparser
import threading
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryWithErrorOutputParser
import ast
import boto3


lock = threading.Lock()

try:
    config = configparser.ConfigParser()
    config.read('conf.properties')

    # __________________ azure credentials __________________

    apiKey = 'XXXXXXXXXXXXXXXxxxxxxxxxxxxx'
    # apiKey = config.get('AzureCredentials', 'apiKey')
    # apiKey = base64.b64decode(apiKey).decode("utf-8")
    EndPoint = config.get('AzureCredentials', 'Endpoint')
    Deployment = config.get('AzureCredentials', 'Deployment')
    version = config.get('AzureCredentials', 'version')
    EmbeddingDeployment = config.get('AzureCredentials', 'EmbeddingDeployment')
    base = EndPoint

    #________________________________________________________ LLM prompts 

    system_prompt = config.get('GenAI', 'system_prompt')
    Rag_prompt = config.get('GenAI', 'Rag_prompt')

    #_____________________________________________________ OS Azure setup
    os.environ["AZURE_OPENAI_ENDPOINT"] = EndPoint
    os.environ["OPENAI_API_KEY"] = apiKey
    open.api_key = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = version
    os.environ["azure_endpoint"] = EndPoint

    #________________________________________________________________ RAG

    rag_topK = int(config.get('Model_parameters', 'rag_topK'))
    rag_threshold = float(config.get('Model_parameters', 'rag_threshold'))

    #____________________________________________________________ available language
    __suported_lang = ast.literal_eval((config.get('Lang', 'supported_lang')))

    #____________________________________________________________ available Models
    __suported_models = config.get('SupporetedModels', 'models').split(',')



except Exception as e:
    print(f"Error in configuration: {str(e)}")
    raise

async def getko(data):
    print(f"Received data: {data}")
    try:
        queries = data.get("data")
        model_name = data.get("model")
        region = data.get("region")
        lang = data.get('lang', None)
        if queries is None or model_name is None or region is None or lang is None:
            return {"error": "Missing one of the parameters - 'data', 'model', 'region' or 'lang'"}, 400
        
        
        if model_name not in __suported_models:
            return {"error": f"No Such Model with name '{model_name}'"}, 400
        
        if lang not in __suported_lang.keys():
            return {"error": "Language not supported"}, 400
        
        
        lang = __suported_lang.get(lang)

        result = await asyncio.gather(
            *[async_from_openai(query, model_name, region, lang) for query in queries]
        )  
        return {"ko_text": result}, 200
    except Exception as e:
        print(f"Error in getko: {str(e)}")
        return {"error": str(e)}, 400

async def async_from_openai(query, model_name, region_name, lang):
    return await asyncio.to_thread(from_openai, query, model_name, region_name, lang)

class KO(BaseModel):
    Query: str = Field(..., description='the root cause of the problem')
    Symptoms: str = Field(..., description='Symptoms refer to the observable signs, behaviors, or effects that indicate the presence of a problem/issue')
    Short_description: str = Field(..., description='A brief and short description of user query')
    Long_description: str = Field(..., description='A detailed and good enhanced description of user query')
    Causes: str = Field(..., description='Causes refer to the underlying reasons or factors that lead to the occurrence of a problem or issue')
    Resolution_note: str = Field(..., description='step by step detailed Enhanced knowledge article which covers all the scenarios')
    Relevancy:str = Field(...,description="Relevancy of the Knowledge article between [0-100]% ")
    
    def get_Ko(self, queryid):
        return {
            'ticketid': queryid,
            'short_description': self.Short_description.replace('\n', '<br>'),
            'long_description': self.Long_description.replace('\n', '<br>'),
            'symptoms': self.Symptoms.replace('\n', '<br>'),
            'causes': self.Causes.replace('\n', '<br>'),
            'resolution_note': self.Resolution_note.replace('\n', '<br>'),
            'Relevancy':self.Relevancy.split("%")[0]
        }

def from_openai(query, model_name, region_name="us-east-1", lang='english'):
    try:
        ticketid = query.get('ticketid', None)
        query = query.get('inc',"").strip()
        if query in ["",None]:
            return {
                "ticketid": ticketid,
                "query": f"unprocessable value for query"
            }
        if model_name.split("@")[-1] == "aws":
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1"
            )
            llm = BedrockChat(model_id=model_name.split("@")[0], client=client,model_kwargs={"max_tokens": 1500})
        else:
            llm = AzureChatOpenAI(temperature=0.5,
                                deployment_name=Deployment,
                                openai_api_version=version,
                                openai_api_key=apiKey, 
                                azure_endpoint=EndPoint)
                                
        template1 = f"""you are an IT assistant for Knowledge article generation,you Must give following field for user query; 
                        short description ,
                        detailed long description,
                        causes(underlying main reasons or factors that lead to the occurrence of a problem),
                        Resolution_note(step by step detailed Enhanced kowledge article which covers all the scenerio),
                        symptoms(observable signs indicate the presence of a problem/issue), 
                        you must provide only one object of 'KO' class where each field is provided,"""
        template2 = """user query: {query},strictly follow specified format only {format_instructions} ,give Resolution_note step by step, """
        template = template1 + template2 + f'response must be in {lang} language only and each fields values must be provided, here is KO object: '
        
        parser = PydanticOutputParser(pydantic_object=KO)
        retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=5)
        prompt = PromptTemplate(
                template=template,
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        completion_chain = prompt | llm
        main_chain = RunnableParallel(completion=completion_chain, prompt_value=prompt) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(x['completion'].content, x['prompt_value']))
        gc.collect()                                
        tries = int(config.get('Tries', 'no_of_calls'))
        while tries >= 1:
            try:
                x = main_chain.invoke({'query': query}).get_Ko(ticketid)
                return x
            except Exception as e:
                tries -= 1
                print(f"Failed to parse values, {tries} tries left. Error: {str(e)}")
        raise Exception("Failed to parse the values after all retries")
    except Exception as e:
        print(f"Error in from_openai: {str(e)}")
        return {"error": str(e)}

async def async_lambda_handler(event, context):
    try:
        event = json.loads(event['body'])
        result, status_code = await getko(event)
        return json.dumps(result), status_code
    except json.JSONDecodeError:
        print("Invalid JSON in event body")
        return json.dumps({"error": "Invalid JSON in request body"}), 400
    except Exception as e:
        print(f"Unexpected error in async_lambda_handler: {str(e)}")
        return json.dumps({"error": "An unexpected error occurred"}), 500

def lambda_handler(event, context):
    try:
        result, status_code = asyncio.run(async_lambda_handler(event, context))
        return {
            'statusCode': status_code,
            'body': result,
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        print(str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "An internal server error occurred"}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
