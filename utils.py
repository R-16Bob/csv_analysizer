import json
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。

1. 对于文字回答的问题，按照这样的格式回答：
   Final Answer: {"answer": "<你的答案写在这里>"}
例如：
   Final Answer: {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   Final Answer: {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   Final Answer: {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   Final Answer: {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   Final Answer: {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。


请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}
(注意不要生成Observation，每次只需要生成Action和Action Input，待工具返回结果再生成Final Answer；其中Action只包含工具的名字，Action Input只包含一行代码，直到能够给出Final Answer）
你要处理的用户请求如下： 
"""

def dataframe_agent(df,query,model="gpt-4o-mini"):
    if model=="gpt-4o-mini":
        model = ChatOpenAI(model="gpt-4o-mini", base_url="https://api.aigc369.com/v1",
                           temperature=0)
    elif model=="glm-4.5":
        model= ChatOpenAI(
            model="glm-4.5",
            openai_api_key=os.getenv("ZAI_API_KEY"),
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0
        )
    else:
        return None
    agent = create_pandas_dataframe_agent(llm=model, df=df,allow_dangerous_code=True,verbose=True)
                                  # agent_executor_kwargs={"handle_parsing_errors":True, "verbose":True})

    response = agent.invoke({"input": PROMPT_TEMPLATE + query})
    response_dict = json.loads(response["output"])
    return response_dict

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("house_price.csv")
    query = "表格中装修状态有哪些？"
    response_dict = dataframe_agent(df,query)
    print(response_dict)
