import streamlit as st
from video_card import video_card
# ========== MAIN CONTENT ==========
def page_ml_basics():
    st.title("📖 Nhập môn Machine Learning")
    st.info("Lộ trình dành cho anh em mới học")

    st.markdown("---")
    st.subheader("🎤  Q&A Khởi Động (3 video)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(video_card("🤔 ML có cần giỏi toán?",
                               "Video Q&A giúp anh em bớt sợ toán.",
                               "https://youtu.be/4DU5WEXQAV4"), unsafe_allow_html=True)

    with col2:
        st.markdown(video_card("🐍 ML cần giỏi code?",
                               "Giúp anh em tự tin nếu mới học Python.",
                               "https://youtu.be/gsOcT2K1q0Y"), unsafe_allow_html=True)

    with col3:
        st.markdown(video_card("🚀 2h Nhập môn ML",
                              "Ngày mình mới bắt đầu học ML (rất Gà)",
                              "https://youtu.be/n3EU_T8pjuM"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📚 Lý Thuyết Cơ Bản (2 video)")

    col4, col5 = st.columns(2)
    with col4:
        st.markdown(video_card("ML là gì?",
                               "Khái niệm cơ bản, ML khác lập trình truyền thống thế nào.",
                               "https://youtu.be/T3Az85XpyUo"), unsafe_allow_html=True)

    with col5:
        st.markdown(video_card("📊 Data, Feature, Label là gì?",
                               "Làm rõ các khái niệm quan trọng trước khi code.",
                               "https://youtu.be/etxB49bmtEU"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛠️ 5 Project Thực Hành (Code ngay)")

    col6, col7 = st.columns(2)
    with col6:
        st.markdown(video_card("🏠 Dự đoán giá nhà với Linear Regression",
                               "Học cách dự đoán giá nhà từ dữ liệu thực tế.",
                               "https://youtu.be/h4zOC4SZb7g"), unsafe_allow_html=True)
    st.markdown("")
    with col7:
        st.markdown(video_card("😊 Dự đoán cảm xúc Logistic Regression",
                               "Nhận diện sentiment từ văn bản.",
                               "https://youtu.be/pIpUTluLr4M"), unsafe_allow_html=True)

    col8, col9 = st.columns(2)
    with col8:
        st.markdown(video_card("📧 Dự đoán email spam Logistic Regression",
                               "Phân loại email spam/ham (không spam).",
                               "https://youtu.be/V3cw3FCCXBk"), unsafe_allow_html=True)

    with col9:
        st.markdown(video_card("🎨 Nén ảnh tạo tranh sơn dầu với KMeans",
                               "Ứng dụng phân cụm để biến ảnh thành tranh sơn dầu.",
                               "https://youtu.be/R7H3iPbwTOk"), unsafe_allow_html=True)

    st.markdown("")
    st.markdown(video_card("🔢 Phân loại chữ số viết tay với KNN",
                           "Nhận dạng số MNIST với K-Nearest Neighbors.",
                           "https://youtube.com/placeholder10"), unsafe_allow_html=True)

    st.markdown("---")
    st.info("👉 Học xong 10 video này, bạn sẽ đủ tự tin để bước vào ML cơ bản.")