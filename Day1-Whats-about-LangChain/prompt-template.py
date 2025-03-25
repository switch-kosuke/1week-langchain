from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

def res_langchain(user_question: str) -> str:
    # LLMの初期化
    # from langchain_ollama import ChatOllama
    # llm = ChatOllama(model = "llama3")
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # プロンプトテンプレートの作成
    template = PromptTemplate.from_template("あなたは{role}です。次の質問に答えてください。{prompt}")

    # アクションの連鎖（Chain）
    chain = template | llm

    res = chain.invoke(input = {"role": "AIアシスタント", "prompt":user_question})
    return res.content

if __name__=="__main__":
    Question = "なぜ空は青いの？"

    # use LangChain
    print(res_langchain(Question))