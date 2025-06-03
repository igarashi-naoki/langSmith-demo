import openai
from langsmith import traceable
from langsmith import Client
import langsmith as ls
from langsmith.wrappers import wrap_openai
import time

client = wrap_openai(openai.Client())
langsmith_client = Client()

# Config used for this example

langsmith_project = "first-langchain-demo"


session_id = "thread-id-1"


langsmith_extra={"project_name": langsmith_project, "metadata":{"session_id": session_id}}

# gets a history of all LLM calls in the thread to construct conversation history

def get_thread_history(thread_id: str, project_name: str): # Filter runs by the specific thread and project
    filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))' # Only grab the LLM runs
    runs = [r for r in langsmith_client.list_runs(project_name=project_name, filter=filter_string, run_type="llm")]

    # Sort by start time to get the most recent interaction
    runs = sorted(runs, key=lambda run: run.start_time, reverse=True)
    # The current state of the conversation
    return runs[0].inputs['messages'] + [runs[0].outputs['choices'][0]['message']]

# if an existing conversation is continued, this function looks up the current run’s metadata to get the session_id, calls get_thread_history, and appends the new user question before making a call to the chat model

@traceable(name="Chat Bot")
def chat_pipeline(question: str, get_chat_history: bool = False): # Whether to continue an existing thread or start a new one
    if get_chat_history:
        run_tree = ls.get_current_run_tree()
        print(f"Current run tree: {run_tree}")
        if run_tree:
            messages = get_thread_history(run_tree.extra["metadata"]["session_id"], run_tree.session_name)
        else:
            messages = []
        messages += [{"role": "user", "content": question}]
    else:
        messages = [{"role": "user", "content": question}]

    # Invoke the model
    print(f"Invoking chat model with messages: {messages}")
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return chat_completion.choices[0].message.content

# Start the conversation

chat_pipeline("Hi, my name is Bob", langsmith_extra=langsmith_extra)

time.sleep(5)

chat_pipeline("What is my name?", get_chat_history=True, langsmith_extra=langsmith_extra)

time.sleep(5)

chat_pipeline("What was the first message I sent you", get_chat_history=True, langsmith_extra=langsmith_extra)


