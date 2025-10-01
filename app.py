import streamlit as st
from dashboard import dash
from page_ml_basics import page_ml_basics
from page_data_science import data_science_page
from page_ml_level_2 import machine_learning_page
from page_deep_learning import deep_learning_page
from page_reinforcement_learning import reinforcement_learning_page
from page_color import page_color


# ========== CONFIG ==========
st.set_page_config(
    page_title = "Gà AI – Hội Viên",
    layout = "wide",
    page_icon="🐣"
)

# ========== SIDEBAR ==========
st.sidebar.title("📚 MENU")
menu = st.sidebar.radio(
    "Điều hướng",
    (
        "🏠 Dashboard",
        "📖 Nhập môn ML",
        "⚡ ML cơ bản",
        "📊 Data Science",
        "🧠 Deep Learning",
        "🎮 Reinforcement Learning",
    )
)

# ========== MAIN CONTENT ==========
if menu == "🏠 Dashboard":
    page_color(color = None)
    dash()

elif menu == "📖 Nhập môn ML":
    page_ml_basics()

elif menu == "⚡ ML cơ bản":
    machine_learning_page()

elif menu == "📊 Data Science":
    data_science_page()

elif menu == "🧠 Deep Learning":
    deep_learning_page()

elif menu == "🎮 Reinforcement Learning":
    reinforcement_learning_page()
