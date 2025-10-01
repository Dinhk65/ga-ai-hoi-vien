import streamlit as st

def page_color(color = '#f0f2f5'):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
