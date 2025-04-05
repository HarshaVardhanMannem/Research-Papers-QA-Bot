"""Chat prompt templates for the QA system."""
from langchain.prompts import ChatPromptTemplate
from config.settings import SYSTEM_MESSAGE

def create_chat_prompt():
    """Create a chat prompt template for the QA system."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ('user', '{input}')
    ])