## ForteBot

ForteBot is an intelligent chatbot designed to provide context-aware responses using advanced natural language processing (NLP) techniques. The project integrates a SQLite database for session tracking and chat history, along with a Flask-based backend for API handling.

### Features

1. Session Management: Tracks user sessions using unique session IDs.

2. Context-Aware Responses: Retrieves relevant documents and generates responses based on context.

3. SQLite Database Integration: Stores chat history and session data.

4. Error Handling: Provides meaningful error messages for various edge cases.

5. Custom Embedding Search: Uses FAISS for efficient similarity search and document retrieval.


### Requirements

Python 3.8 or higher
Flask
SQLite
FAISS (Facebook AI Similarity Search)
NumPy

Install dependencies using the following command:
##### pip install -r requirements.txt
