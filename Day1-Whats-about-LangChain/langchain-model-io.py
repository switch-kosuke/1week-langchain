from dotenv import load_dotenv
load_dotenv()

###############
# OpenAI`s API
###############
def res_openai(user_question: str) -> str: 
    # @ref: https://platform.openai.com/docs/quickstart?api-mode=chat&lang=python
    from openai import OpenAI
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": user_question
            }
        ]
    )

    return completion.choices[0].message.content

###############
# Google`s Gemini API
###############
def res_google(user_question: str) -> str: 
    # @ref: https://ai.google.dev/gemini-api/docs/quickstart?hl=ja&lang=python
    from google import genai

    client = genai.Client(api_key="YOUR_API_KEY")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=user_question
    )
    return response.text

###############
# ollama`s Library
###############
def res_ollama(user_question: str) -> str: 
    # @ref: https://github.com/ollama/ollama-python
    from ollama import chat
    from ollama import ChatResponse

    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': user_question,
        },
    ])
    return response['message']['content']
    # or access fields directly from the response object
    #return response.message.content

###############
# LangChain`s LLM API
###############

def res_langchain(user_question: str) -> str:
    # OpenAI API
    # lib: pip install langchain-openai
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o")

    # # Gemini API
    # # lib: pip install -U langchain-google-genai
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # ollama Lib
    # lib: pip install -U langchain-ollama
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model = "llama3")

    res = llm.invoke(user_question)
    return res.content

if __name__=="__main__":
    Question = "なぜ空は青いの？"

    # # use openai
    # print(res_openai(Question))

    # # use gemini
    # print(res_google(Question))

    # # use ollama
    # print(res_ollama(Question))

    # use LangChain
    print(res_langchain(Question))