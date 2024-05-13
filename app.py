import streamlit as st
from langchain import verbose
#from langchain.globals import set_verbose, get_verbose


#current_level = verbose.get_level(2)
#current_level = get_verbose(2)


from langchain_community.llms import Ollama

import pandas as pd
from pandasai import SmartDataframe

llm = Ollama(model="llama3")


st.title("Data Analysis with PandasAI")

uploader_file = st.file_uploader("Upload a CSV file",type =["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))
    df = SmartDataframe(data, config={"llm":llm})
    prompt = st.text_area("Enter your Prompt:")

    if st.button("Generate:"):
        if prompt:
            with st.spinner(("Generative response...")):
                st.write(df.chat(prompt))
        else:
            st.warning("Please enter a prompt!")     