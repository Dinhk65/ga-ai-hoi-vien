import streamlit as st

# ========== MAIN CONTENT ==========
def video_card(title, desc, link):
    return f"""
    <div style="background-color:#fefefe; padding:15px; border-radius:12px;
                box-shadow:1px 1px 4px rgba(0,0,0,0.1); margin-bottom:15px;">
        <h4>{title}</h4>
        <p>{desc}</p>
        <a href="{link}" target="_blank" style="text-decoration:none;">
            <button style="background-color:#ff4b4b; color:white; border:none; 
                           padding:8px 15px; border-radius:8px; cursor:pointer;">
                ▶ Xem video
            </button>
        </a>
    </div>
    """