from fastapi import FastAPI
from pydantic import BaseModel
import math
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate



# Initialize FastAPI app
app = FastAPI()


# Langchain Output Format
response_schemas = [
    ResponseSchema(name="question", description="Question given in the prompt."),
    ResponseSchema(
        name="Answer",
        description="Precise answer to the question",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Prompt Template
template = """
You are a scientiic calculator. You can solve Addition, Subtraction, Multiplication and Division. You can evluate expressions with Trigonometric, log, power and exponents function. 

\n

question: {question}

Take care of the sign convention.

\n

{format_instructions}

"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

# Loading LLM
model = OllamaLLM(model="llama3.1")  # 8b


# Creating Chain
chain = prompt | model | output_parser


# Define the input model
class Expression(BaseModel):
    expression: str

# Define a POST route to evaluate the expression
@app.post("/calculate")
async def calculate(expression: Expression):
    try:

        response = chain.invoke({'question' : expression.expression})
        print(response)
        return {"result": response['Answer']}
    except Exception as e:
        return {"result": "Invalid Expression"}

# Adding CORS middleware (if using a separate frontend)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with the frontend URL if different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
