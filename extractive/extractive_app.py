# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:12:51 2021

@author: Abhinav
"""

import streamlit as st

from extractive_summarizer import (TFIDFExtractiveSummarizer, 
                                   GensimExtractiveSummarizer, 
                                   SumyExtractiveSummarizer)

if __name__ == '__main__':


    st.header("Extractive Text Summarization")
    st.subheader("This app will summarize the long piece of input text in a few sentences using various algorithms/ways.")

    st.subheader("Paste your raw text below:")
    text = st.text_area(label="Input Text")
    if st.button("Summarize"):
        if text:
            summarizer = TFIDFExtractiveSummarizer()
            summary = summarizer.get_summary(text)
            st.subheader("Using TF-IDF Algorithm:")
            st.success(summary)

            summarizer = GensimExtractiveSummarizer()
            summary = summarizer.get_summary(text)
            st.subheader("Using Gensim Summarizer API:")
            st.success(summary)
            
            summarizer = SumyExtractiveSummarizer()
            summary = summarizer.get_summary(text)
            st.subheader("Using Sumy Summarizer API:")
            st.success(summary)
        else:
            st.error("Please paste or write(!) some text")