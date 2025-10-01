import streamlit as st
from video_card import video_card
from card import card

# ================== PAGE: DATA SCIENCE ==================
def data_science_page():
    st.title("📊 Data Science – Lộ trình toàn diện")
    st.info("Từ thao tác dữ liệu cơ bản đến trực quan hóa & storytelling")

    # ================== CHƯƠNG 1 ==================
    st.markdown("---")
    st.header("📂 Chương 1: Làm Quen Với Dữ Liệu Thực")
    st.info("🎯 Mục tiêu: Nắm định dạng dữ liệu phổ biến, đọc/ghi với Pandas, khám phá sơ bộ")

    with st.expander("Xem chi tiết: "):
            # Bài học
            col1, _, col2 = st.columns([8, 1, 8])
            with col1:
                card("Bài 1: Định dạng dữ liệu phổ biến",
                     "Cấu trúc vs bán cấu trúc vs phi cấu trúc. CSV, JSON, Excel, SQL, API…",
                     icon="📑",
                     color = '#e8f5e9'
                     )
                card("Bài 2: Pandas cơ bản",
                     "Series vs DataFrame, tạo DataFrame từ dict, kiểm tra .shape, .columns, .dtypes",
                     icon="📑",
                     color = '#e8f5e9'
                     )
                card("Bài 3: Đọc dữ liệu",
                     "pd.read_csv(), pd.read_excel(), pd.read_json(). Thực hành với dataset thật.",
                     icon="📑",
                     color = '#e8f5e9'
                     )
                card("Bài 4: Ghi dữ liệu ra file",
                     "Xuất dữ liệu sang CSV, Excel, JSON với to_csv(), to_excel()",
                     icon="📑",
                     color = '#e8f5e9'
                     )
            with col2:
                card("Bài 5: Khám phá dữ liệu",
                     ".info(), .describe(), ý nghĩa mean, std, min, max, count",
                     icon="📑",
                     color = '#e8f5e9'
                     )
                card("Bài 6: Làm việc với columns/index",
                     "Truy cập, đổi tên cột, lọc bằng .loc[], .iloc[]",
                     icon="📑",
                     color = '#e8f5e9'
                     )
                card("Bài 7: Dữ liệu thiếu",
                     "Xử lý NaN với .isnull(), .dropna(), .fillna()",
                     icon="📑",
                     color = '#e8f5e9'
                     )
            card("Mini Project 1: COVID-19",
                     "Đọc dữ liệu Our World In Data, phân tích sơ bộ và lưu bản rút gọn.",
                     icon="📋",
                     color = 'linear-gradient(135deg, #fbc2eb, #a6c1ee)'
                     )

            card("Mini Project 2: Dân số & GDP",
                 "Dữ liệu World Bank, lọc 5 quốc gia GDP cao nhất, lưu kết quả.",
                  icon="📋",
                  color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
                 )

    # ================== CHƯƠNG 2 ==================
    st.markdown("---")
    st.header("🧹 Chương 2: Data Cleaning")
    st.info("🎯 Mục tiêu: Làm sạch dữ liệu thiếu, trùng lặp, sai định dạng, chuẩn hóa")

    # ================== CHƯƠNG 2: DATA CLEANING ==================
    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Giới thiệu về Data Cleaning",
                 "Tầm quan trọng trong pipeline Data Science. Các vấn đề: Missing, Duplicate, Invalid, Inconsistent. Công cụ: pandas, numpy, regex.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Phát hiện dữ liệu thiếu",
                 "Dùng .isnull(), .info(), .sum(). Trực quan hóa với missingno, seaborn. Phân loại MCAR, MAR, MNAR.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Chiến lược xử lý Missing",
                 "Xóa bằng .dropna(), điền giá trị với mean/median/mode, forward-fill, backward-fill, hoặc mô hình (KNN/Regression).",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 4: Dữ liệu trùng lặp",
                 "Phát hiện bằng .duplicated(), xử lý với .drop_duplicates(), cân nhắc theo mục tiêu phân tích.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 5: Dữ liệu sai định dạng & lỗi logic",
                 "Phát hiện lỗi bằng regex, str.contains(), apply(). Ví dụ: Email, số điện thoại, tuổi âm, ngày tháng sai.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Chuẩn hóa dữ liệu",
                 "Chuẩn hóa chữ (.str.lower(), .str.strip()), datetime (pd.to_datetime), đổi dtype, mapping dữ liệu không nhất quán.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Mini Projects
        card("Mini Project 1: Làm sạch dữ liệu khách hàng",
             "Dataset khách hàng: xử lý thiếu tuổi/email, xóa trùng, chuẩn hóa tên, giới tính, sdt, validate email.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Mini Project 2: Làm sạch dữ liệu giao dịch",
             "Dataset ngân hàng: làm sạch datetime, xử lý giá trị âm, kiểm tra trùng lặp, chuẩn hóa loại giao dịch.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 3 ==================
    st.markdown("---")
    st.header("🛠️ Chương 3: Feature Engineering")
    st.info(
        "🎯 Mục tiêu: Hiểu & áp dụng các kỹ thuật Feature Engineering từ cơ bản đến nâng cao để tối ưu đầu vào mô hình ML.")

    # ================== CHƯƠNG 3: FEATURE ENGINEERING ==================
    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Tổng quan Feature Engineering",
                 "Feature Engineering trong pipeline ML. Phân loại biến: numerical, categorical, datetime, text, ordinal, binary. Tầm quan trọng của domain knowledge.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Làm sạch & chuyển đổi kiểu dữ liệu",
                 "Chuyển đổi kiểu với .astype(). Sửa lỗi khi ép kiểu. Category vs object. Tự động profiling kiểu dữ liệu.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Tạo biến mới (Feature Construction)",
                 "Tạo biến toán học, logic, tổng hợp. Feature từ datetime. Kỹ thuật nâng cao: Interaction, Polynomial, Crossed Features.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Encoding dữ liệu phân loại",
                 "Cơ bản: Label, One-hot. Nâng cao: Target, Binary, Frequency, CatBoost Encoding. Ưu/nhược điểm theo mô hình.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 5: Scaling & Transformation",
                 "Cơ bản: Min-Max, StandardScaler. Nâng cao: RobustScaler, Power Transform (Box-Cox, Yeo-Johnson), Log-transform.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 6: Outlier – Phát hiện & xử lý",
                 "Phát hiện bằng IQR, Z-score, Isolation Forest, LOF. Chiến lược: clipping, imputation, log, winsorizing. Phân biệt kỹ thuật vs thực tế.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Binning (Phân nhóm)",
                 "Fixed-width, Quantile bins. Kỹ thuật nâng cao: K-means binning, Decision tree binning, Weight of Evidence.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Xử lý dữ liệu thời gian",
                 "Trích xuất feature từ timestamp (ngày, tháng, mùa, giờ). Tính khoảng thời gian. Cyclical encoding với sin/cos.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 9: Feature Selection & Importance",
                 "Filter: Variance, Correlation. Wrapper: RFE. Embedded: Feature Importance, Lasso. Giải thích với SHAP, Permutation.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Project nâng cao
        card("Project Nâng Cao: Feature Engineering Toàn Tập",
             "Dataset khách hàng mở rộng: tạo RFM từ giao dịch, Target Encoding occupation, xử lý outlier với IQR + RobustScaler, K-means binning thu nhập, trích xuất datetime, phân tích SHAP.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 4 ==================
    st.markdown("---")
    st.header("📊 Chương 4: Thống Kê Mô Tả (Descriptive Statistics)")
    st.info(
        "🎯 Mục tiêu: Hiểu & áp dụng các kỹ thuật thống kê mô tả để khám phá dữ liệu, nắm bắt đặc điểm cơ bản và hỗ trợ phân tích EDA."
    )

    # ================== CHƯƠNG 4: DESCRIPTIVE STATISTICS ==================
    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card(
                "Bài 1: Giới thiệu Thống kê Mô tả",
                "Tại sao thống kê mô tả là bước đầu tiên? Vai trò trong EDA. Phân biệt dữ liệu định lượng vs định tính.",
                icon="📑",
                color="#e8f5e9"
            )
            card(
                "Bài 2: Thống kê Trung tâm (Central Tendency)",
                "Mean, Median, Mode. Nâng cao: Trimmed Mean, Weighted Mean. So sánh khi dữ liệu lệch hoặc có outlier.",
                icon="📑",
                color="#e8f5e9"
            )
            card(
                "Bài 3: Phương sai & Độ lệch chuẩn",
                "Ý nghĩa variance & std. Tính bằng pandas (.var(), .std()). Nâng cao: Coefficient of Variation, group std().",
                icon="📑",
                color="#e8f5e9"
            )
            card(
                "Bài 4: Phân phối dữ liệu",
                "Histogram, density plot. Skewness (độ lệch), Kurtosis (độ nhọn). Nâng cao: kiểm định normality với scipy.stats.",
                icon="📑",
                color="#e8f5e9"
            )

        with col2:
            card(
                "Bài 5: Tần suất & Tỷ lệ",
                "value_counts(), tính phần trăm, bảng tần suất. Nâng cao: Cross-tab (pd.crosstab), bar chart, pie chart.",
                icon="📑",
                color="#e8f5e9"
            )
            card(
                "Bài 6: Tổng hợp theo nhóm",
                "groupby() cơ bản: mean, count, std. pivot_table() để tổng hợp linh hoạt. Nâng cao: agg(), apply(), multi-index group.",
                icon="📑",
                color="#e8f5e9"
            )
            card(
                "Bài 7: Phân tích theo chiều thời gian",
                "Tổng hợp theo tháng/quý/năm. Rolling mean/std. Nâng cao: resample(), seasonal pattern với line plot, heatmap.",
                icon="📑",
                color="#e8f5e9"
            )

        # Project thực hành
        card(
            "Project Thực Hành: Phân tích Khách Hàng",
            "Dataset khách hàng: tính thu nhập trung bình theo gender/occupation/age group, phân phối thu nhập + skewness, cross-tab ngành nghề vs giới tính, tổng hợp chi tiêu theo tháng.",
            icon="📋",
            color = 'linear-gradient(135deg, #fbc2eb, #a6c1ee)'
        )

    # ================== CHƯƠNG 5 ==================
    st.markdown("---")
    st.header("📈 Chương 5: Trực Quan Hóa Dữ Liệu")
    st.info(
        "🎯 Mục tiêu: Biết chọn loại biểu đồ phù hợp, làm chủ Matplotlib & Seaborn, mở rộng sang Plotly/Altair để kể chuyện thuyết phục bằng dữ liệu và xây dựng dashboard."
    )

    # ================== CHƯƠNG 5: DATA VISUALIZATION ==================
    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Kể chuyện bằng biểu đồ (Data Storytelling)",
                 "Chọn biểu đồ theo mục tiêu (phân phối, xu hướng, mối quan hệ, phân loại). Hiểu cấu trúc 1 biểu đồ tốt. Tips: annotate, highlight, tránh lạm dụng màu.",
                 icon="📑",
                 color="#e8f5e9"
                 )
            card("Bài 2: Custom nâng cao với Matplotlib & Seaborn",
                 "Tùy chỉnh ticks, legend, subplot phức tạp. Thêm annotation (trend line, vùng cảnh báo). Chuyên sâu: seaborn-whitegrid, Facet, basemap.",
                 icon="📑",
                 color="#e8f5e9"
                 )
            card("Bài 3: Trực quan hóa mối quan hệ đa biến",
                 "Pairplot, Jointplot, Correlation Heatmap. Mối quan hệ 3 biến với size, color, style. Pro: lmplot đa lớp, hexbin, density contour.",
                 icon="📑",
                 color="#e8f5e9"
                 )

        with col2:
            card("Bài 4: Visualization theo thời gian",
                 "Lineplot, rolling average, seasonality, multi-line, area chart. Nâng cao: event markers, resample() + trực quan.",
                 icon="📑",
                 color="#e8f5e9"
                 )
            card("Bài 5: Trực quan hóa động (Interactive Visualization)",
                 "Plotly: px.line, px.scatter, px.bar với hover tooltips. Altair: filter, selection, dashboard mini với Streamlit. Pro: drill-down dashboard.",
                 icon="📑",
                 color="#e8f5e9"
                 )
            card("Bài 6: Visualization cho phân tích mô hình",
                 "ROC, Precision-Recall, Feature Importance (bar, SHAP), Residual plots. Đánh giá trực quan kết quả mô hình.",
                 icon="📑",
                 color="#e8f5e9"
                 )

        # Case Study
        card("Case Study: Phân tích trực quan toàn diện",
             "Dataset khách hàng & chi tiêu theo thời gian: phân phối thu nhập (age, gender), heatmap tương quan, line chart chi tiêu theo tháng, violin plot nghề nghiệp, dashboard tương tác.",
             icon="📋",
             color="linear-gradient(135deg, #fbc2eb, #a6c1ee)"
             )

    st.markdown("---")
    st.success("👉 Học xong Data Science track, bạn sẽ thành thạo Numpy, Pandas, Seaborn, biết xử lý & kể chuyện với dữ liệu thực tế.")