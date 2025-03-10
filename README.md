## Vitafy AI Chatbot
@app.py

Overview This documentation provides insights into a FastAPI application designed for handling document uploads, processing web URLs, managing Q&A pairs, and storing data in a Pinecone vector store. The application leverages OpenAI models for generating embeddings and responses. Required Libraries The application uses several libraries, including but not limited to:
1. langchain
2. langchain-community
3. pinecone
4. ChatOpenAI
5. langsmith
6. fastapi
7. unstructured
8. RetrievalQA
9. RecursiveCharacterTextSplitter
10. PineconeHybridSearchRetriever
11. openai

# Application Configuration
FastAPI Initialization The FastAPI application is initialized with the following parameters:

CORS Middleware CORS is enabled for all origins, allowing cross-origin requests.
origins = ["*"]
app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"])

# Environment Variables
Environment variables are loaded using dotenv, which includes keys for OpenAI and Pinecone. If the Pinecone API key is missing, the application prompts the user to enter it.
Pinecone Initialization
Checks for existing Pinecone indexes and creates one if it does not exist, specifying dimensions and metrics.
# API Endpoints
1. File Upload 
Endpoint: POST /api/artificial-intelligence/upload

Request Body
tenantId: Required tenant identifier.
files: A list of files to upload (PDF, DOCX, TXT).

Functionality
Validates file types and detects duplicates.
Loads the documents and splits them into chunks.
Generates embeddings and stores them in Pinecone
Response body
{
"status":"success",
"message":"Files uploaded successfully."
}

2. Web URL Upload
Endpoint: POST /api/artificial-intelligence/links

Request Body
tenantId: Required tenant identifier.
urls: A list of URLs to process.
Functionality
Loads content from provided URLs.
Processes the content and embeddings and stores it in Pinecone.

Response body
{
"status":"success",
"message":"URLs processed and stored successfully."
}

3. Question and Answer Upload
Endpoint: POST /api/artificial-intelligence/qa

Request Body
tenantId: Required tenant identifier.
question: A list of questions.
answer: A list of answers.
Functionality
Processes the question-answer pairs.
Generates embeddings and stores them in Pinecone

Response body
{
"status":"success",
"message":"Question and Answer processed  successfully.",
"id":"b566aa6b-8cd0-4840-b93e-604171df8ca1",
"question":"what is your name?",
"answer":"my name is ujjwal khadka and i m an ai engineer"
}

4. Retrieve Tenant Files Endpoint
Endpoint: GET /api/artificial-intelligence/tenant_files

Parameters
tenantId: Required tenant identifier.

Functionality
Retrieves all uploaded documents associated with the given tenant ID.
Response body
{
"data": [
    {
"fileName":"Billings Module.docx",
"id":"e71b5108-9672-4f74-8901-c180efd86aa2",
"tenantId":"tenant1"
    }
  ]
}

5. Prompt Keyword
Endpoint: GET /api/artificial-intelligence/prompts

Parameters
tenantId: Required tenant identifier.
keyword: Keyword for search queries.

Functionality
Searches for documents related to the provided keyword and retrieves relevant responses.

Response body
To issue a refund, you can follow these steps:

1. Identify the transaction you want to refund.
2. Use the ellipses ⋮ on the far right of the transaction.
3. Choose the refund option.
4. Follow the prompts to complete the refund back to the original form of payment.

6. Document Deletion 
Endpoint: DELETE /api/artificial-intelligence/delete

Request Body
prefix: Prefix string to filter the documents to delete.
Functionality
Deletes all documents in the Pinecone index that match the given prefix. Running the Application
Response body
{
"message":"Deleted 1 documents with prefix 'e71b5108-9672-4f74-8901-c180efd86aa2'."
}

7. Edit QA
Endpoint: PUT /api/artificial-intelligence/qa

8. Delete QA
Endpoint: DELETE /api/artificial-intelligence/qa


To run the application, execute the following command:
uvicorn main:app --host 0.0.0.0 --port 8000

This command starts the FastAPI server on the specified host and port.
Conclusion This application provides a comprehensive interface for managing document uploads, web content processing, and Q&A storage, leveraging the capabilities of OpenAI and Pinecone for enhanced data handling and retrieval.