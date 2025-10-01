import streamlit as st

def dash():
    _, col_title, _ = st.columns([2, 5, 1])
    with col_title:
        st.title("🐔 Hub Hội Viên – Gà AI")
    st.info("Chào mừng hội viên đến với trung tâm học tập AI/ML cùng Gà AI")

    st.markdown("---")

    # ====== CARD LAYOUT ======
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
                <div style="background-color:#fce4ec; padding:20px; border-radius:15px;
                            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                    <h3>📖 Nhập môn ML</h3>
                    <p>Bắt đầu với những khái niệm cơ bản kèm 5 project mini để nhập môn.</p>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
                <div style="background-color:#fce4ec; padding:20px; border-radius:15px;
                            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                    <h3>⚡ ML cơ bản</h3>
                    <p>Làm quen với các thuật toán Machine Learning nền tảng và project từ A đến Z.</p>
                </div>
                """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
                <div style="background-color:#fce4ec; padding:20px; border-radius:15px;
                            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                    <h3>📊 Data Science</h3>
                    <p>Khám phá dữ liệu, trực quan hóa, xử lý dữ liệu thực tế.</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("""
                <div style="background-color:#fce4ec; padding:20px; border-radius:15px;
                            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                    <h3>🧠 Deep Learning</h3>
                    <p>Tìm hiểu mạng nơ-ron,MLP, CNN, RNN, LSTM, Transformer.</p>
                </div>
                """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
                <div style="background-color:#fce4ec; padding:20px; border-radius:15px;
                            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                    <h3>🎮 Reinforcement Learning</h3>
                    <p>Học nền tảng về học tăng cường từ A đến Z.</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 Tip: Chọn mục ở sidebar để bắt đầu học!")