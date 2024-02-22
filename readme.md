### Prerequisite
You must acquire an OpenAI API key to start the project. \
Set the api key in a **.env** file \
`OPENAI_API_KEY="sk-<YOUR-API-KEY>"`


#### Steps to run the app
Install Python v3.9 to v3.11

##### From the project dir.
Create a virtual environment: \
`python -m venv venv`

*In case of windows shows permissions error*
\
Run `Set-ExecutionPolicy Unrestricted -Scope Process`

Activate the environment: \
`.\venv\Scripts\activate`

Install the following dependencies: \
`pip install python-dotenv langserve[all] langchain==0.1.6 langchain-community==0.0.19 langchain-core==0.1.23 langchain_openai python-multipart pypdf docx2txt faiss-cpu`

For parsing from web: \
`pip install beautifulsoup4`


