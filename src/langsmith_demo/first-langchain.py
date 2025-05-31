from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import LangChainTracer

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "日本で一番大きい都市はどこですか？"
context = "During this morning's meeting, we solved all world conflict."

tracer = LangChainTracer(project_name="first-langchain-demo")

chain.invoke({"question": question, "context": context}, config={"callbacks": [tracer],"tags": ["invoke-tag"], "metadata": {"invoke-key": "invoke-value"}, "run_name": "MyCustomChain"}) 