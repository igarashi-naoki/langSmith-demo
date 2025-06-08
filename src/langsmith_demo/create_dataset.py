from langsmith import Client

client = Client()

# Programmatically create a dataset in LangSmith
# For other dataset creation methods, see:
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application
dataset = client.create_dataset(
    dataset_name="Sample dataset", description="A sample dataset in LangSmith."
)

# Create examples
examples = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
    },
]

# Add examples to the dataset
client.create_examples(dataset_id=dataset.id, examples=examples)