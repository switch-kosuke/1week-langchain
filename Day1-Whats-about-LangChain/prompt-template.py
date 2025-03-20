from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def res_langchain(user_question: str) -> str:
    # LLMの初期化
    llm = ChatOllama(model = "llama3")

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