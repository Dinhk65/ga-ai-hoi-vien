import streamlit as st

# ===== Card Component =====
def card(title, desc, link=None, icon="📌", color = 'linear-gradient(135deg, #ffffff, #f8f9ff)'):
    st.markdown(
        f"""
        <div style="
            padding:1rem;
            border-radius:12px;
            background:{color};
            margin-bottom:1rem;
            box-shadow:0 2px 6px rgba(0,0,0,0.08);
        ">
            <h3 style="margin:0;">{icon} {title}</h3>
            <p style="margin:0.2rem 0 0.6rem 0; color:#444;">{desc}</p>
            {f'<a href="{link}" target="_blank" style="text-decoration:none; color:#0066cc; font-weight:600;">👉 Visit</a>' if link else ""}
        </div>
        """,
        unsafe_allow_html=True
    )
# """
# Gọi hàm:
#
# card(
#     "Model",
#     "Logistic Regression",
#     link="https://colab.research.google.com/drive/1uL0i0KvAPOujnt0WUkjAlc-oQcgFzqz9#scrollTo=l8J10ix7RFW7",
#     icon="🤖"
# )
# """
