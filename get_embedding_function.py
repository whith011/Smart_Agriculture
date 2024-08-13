from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings  # Ensure correct import path
import logging


def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="eu-west-2",
        model_id="amazon.titan-embed-text-v2:0"  # Correct parameter
    )
    return embeddings
