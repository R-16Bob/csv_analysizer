import pandas as pd
import streamlit as st
from utils import dataframe_agent

st.title("💡CSV数据分析工具")

def create_chart(input_data,chart_type):
    df_data = pd.DataFrame(input_data["data"],columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0],inplace=True) # 把第一列设为索引，即图标的横轴
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)
# 添加模型选择下拉框
model_choice = st.selectbox("请选择模型", ["gpt-4o-mini", "glm-4.5"])
data = st.file_uploader("请上传你的CSV文件：",type=["csv"])

if data:
    st.session_state['df'] = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state['df'])
query = st.text_area("请输入你对以上表格的问题，或提取请求，或可视化请求（支持散点图、折线图、条形图）")
button= st.button("生成回答")
if button and 'df' not in st.session_state:
    st.info("请先上传数据文件")
elif button and 'df' in st.session_state:
    with st.spinner("AI正在思考中..."):
        response_dict = dataframe_agent(st.session_state['df'],query, model=model_choice)
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        elif "table" in response_dict:
            st.dataframe(pd.DataFrame(response_dict["table"]["data"], columns=response_dict["table"]["columns"]))
        elif "bar" in response_dict:
            create_chart(response_dict["bar"],"bar")
        elif "line" in response_dict:
            create_chart(response_dict["line"],"line")
        elif "scatter" in response_dict:
            create_chart(response_dict["scatter"],"scatter")
