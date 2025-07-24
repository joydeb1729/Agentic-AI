
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from fastapi import FastAPI
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes


# 1. Create Prompt Template
system_message = "You give concise answer and only Translate the following text to : {language}"

promt_template = ChatPromptTemplate.from_messages(
    [("system", system_message), 
     ("human", "{text}")]
)

#2. Create LLM
llm = ChatGroq(model="llama3-8b-8192")

#3. Create Output Parser
parser = StrOutputParser()

#4. Create Chain
chain = promt_template | llm | parser


app = FastAPI(
    title="Langserve Demo",
    description="A simple demo of Langserve with Groq LLM",
    version="0.1.0",
)    

add_routes(
    app,
    chain,
    path="/translate"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    # To run the server, use the command: 