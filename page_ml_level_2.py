import streamlit as st
from video_card import video_card
from card import card

# ================== PAGE: MACHINE LEARNING ==================
def machine_learning_page():
    st.title("🤖 Machine Learning – Lộ trình từ cơ bản đến nâng cao")
    st.info("Từ thuật toán cơ bản đến ứng dụng thực tế trong dự án ML")

    # ================== CHƯƠNG 1 ==================
    st.markdown("---")
    st.header("🚀 Chương 1: Nhập Môn Machine Learning")
    st.info("🎯 Mục tiêu: Hiểu bản chất ML, thuật toán cơ bản, cách đánh giá và tối ưu mô hình")

    with st.expander("Xem chi tiết: "):
        # Bài học
        col1, _, col2 = st.columns([8, 1, 8])
        with col1:
            card("Bài 1: Machine Learning là gì?",
                 "Phân biệt AI, ML, DL. Supervised vs Unsupervised vs Reinforcement. Workflow ML: Data → Feature → Model → Evaluation → Deploy.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Hồi quy tuyến tính (Linear Regression)",
                 "Công thức y = wx + b. Tìm w, b tối ưu bằng Least Squares. Thực hành với scikit-learn: fit(), predict(), score().",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Hồi quy logistic (Logistic Regression)",
                 "Sigmoid function cho phân loại nhị phân. Probability vs Binary prediction. Thực hành phân loại khách hàng mua/không mua.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Loss Functions",
                 "MSE cho regression, Cross-Entropy cho classification. Hiểu tại sao chọn loss function phù hợp với từng bài toán.",
                 icon="📑",
                 color='#e8f5e9'
                 )
        with col2:
            card("Bài 5: Gradient Descent chuyên sâu",
                 "Cách thuật toán 'học' bằng cách giảm loss. Learning rate, epoch, batch size. Stochastic vs Mini-batch vs Full-batch GD.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Train/Validation/Test Split",
                 "Tại sao cần chia dữ liệu? Cross-validation (K-fold). Thực hành với train_test_split(), KFold().",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Đánh giá mô hình",
                 "Regression: MAE, RMSE, R². Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Preprocessing cho ML",
                 "Chuẩn hóa dữ liệu với StandardScaler, MinMaxScaler. Encoding categorical với LabelEncoder, OneHotEncoder.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        card("Mini Project 1: Dự đoán giá nhà",
             "Dataset Boston Housing: EDA, preprocessing, Linear Regression, đánh giá với RMSE/R², visualize predicted vs actual.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Mini Project 2: Phân loại khách hàng",
             "Dataset Titanic: EDA, feature engineering, Logistic Regression, đánh giá bằng Confusion Matrix, ROC curve.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 2 ==================
    st.markdown("---")
    st.header("🎯 Chương 2: Supervised Learning - Thuật Toán Nâng Cao")
    st.info("🎯 Mục tiêu: Làm chủ các thuật toán ML quan trọng, hiểu ưu nhược điểm và cách áp dụng thực tế")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Hồi quy bậc cao (Polynomial Regression)",
                 "Mở rộng Linear Regression với PolynomialFeatures. Bias-Variance tradeoff. Tìm degree tối ưu bằng validation curve.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Decision Tree - Thuật toán chia nhánh",
                 "Cách tree 'quyết định' bằng Gini Impurity, Information Gain. Hyperparameters: max_depth, min_samples_split. Visualize tree.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Random Forest - Sức mạnh tập thể",
                 "Ensemble của nhiều Decision Tree. Bootstrap Aggregating (Bagging). Feature importance, Out-of-bag score.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Support Vector Machine (SVM)",
                 "Tìm 'đường biên tốt nhất' (optimal hyperplane). Kernel trick: linear, polynomial, RBF. C parameter và gamma.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: Naive Bayes - Xác suất đơn giản",
                 "Bayes' theorem trong ML. GaussianNB, MultinomialNB. Ứng dụng: phân loại email spam, sentiment analysis.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: K-Nearest Neighbors (KNN)",
                 "Thuật toán 'hàng xóm gần nhất'. Chọn k tối ưu, distance metrics (Euclidean, Manhattan). Lazy learning.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Overfitting & Regularization",
                 "Nhận biết overfitting qua learning curve. L1 (Lasso), L2 (Ridge) regularization. Dropout trong Neural Networks.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Hyperparameter Tuning",
                 "Grid Search vs Random Search vs Bayesian Optimization. Cross-validation cho tuning. Pipeline trong scikit-learn.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Mini Projects
        card("Project 1: So sánh thuật toán",
             "Dataset Iris: thử nghiệm Decision Tree, Random Forest, SVM, KNN trên cùng dataset. So sánh accuracy, training time, interpretability.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: Phân loại email spam",
             "Dataset email: text preprocessing, TF-IDF, so sánh Naive Bayes vs SVM. Đánh giá với Precision/Recall vì class imbalance.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 3 ==================
    st.markdown("---")
    st.header("🔍 Chương 3: Unsupervised Learning")
    st.info("🎯 Mục tiêu: Khám phá dữ liệu không nhãn, phân cụm khách hàng, giảm chiều dữ liệu và nén thông tin")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Giới thiệu Unsupervised Learning",
                 "Bài toán không có 'đáp án'. Ứng dụng: customer segmentation, anomaly detection, data compression. So sánh với Supervised Learning.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: K-Means Clustering",
                 "Thuật toán phân cụm phổ biến nhất. Centroids, inertia, elbow method để chọn k. Ưu nhược điểm, assumption về cluster shape.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: DBSCAN - Phân cụm dựa trên mật độ",
                 "Tìm cluster có hình dạng bất kỳ. Parameters: eps, min_samples. Xử lý noise points. So sánh với K-Means.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Hierarchical Clustering",
                 "Agglomerative vs Divisive. Dendrogram để visualize. Linkage criteria: single, complete, average, ward.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: Principal Component Analysis (PCA)",
                 "Giảm chiều dữ liệu bằng cách tìm 'hướng quan trọng nhất'. Eigenvalues, Eigenvectors. Explained variance ratio.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: t-SNE - Visualization đa chiều",
                 "Giảm chiều cho visualization. Perplexity parameter. So sánh với PCA: linear vs non-linear dimensionality reduction.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Autoencoder - Mô hình nén dữ liệu",
                 "Neural Network học cách 'nén' và 'giải nén' dữ liệu. Encoder-Decoder architecture. Ứng dụng: denoising, anomaly detection.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Anomaly Detection",
                 "Phát hiện dữ liệu 'bất thường'. Isolation Forest, One-Class SVM. Ứng dụng: fraud detection, system monitoring.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Projects
        card("Project 1: Customer Segmentation",
             "Dataset khách hàng: RFM analysis, K-Means clustering, PCA visualization, profile từng segment, business insights.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: Phân tích sản phẩm",
             "Dataset sản phẩm e-commerce: clustering sản phẩm tương tự, PCA giảm chiều features, t-SNE visualization, recommendation system đơn giản.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 4: ENSEMBLE & ADVANCED ==================
    st.markdown("---")
    st.header("⚡ Chương 4: Ensemble Methods & Advanced Techniques")
    st.info("🎯 Mục tiêu: Nâng cao hiệu suất mô hình với Ensemble, làm chủ XGBoost và các kỹ thuật ML nâng cao")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Ensemble Learning Overview",
                 "Wisdom of crowds trong ML. Bagging vs Boosting vs Stacking. Bias-Variance tradeoff trong ensemble.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Bagging & Extra Trees",
                 "Random Forest chi tiết hơn. ExtraTreesClassifier. Out-of-bag evaluation. Parallel processing trong ensemble.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Boosting - AdaBoost & Gradient Boosting",
                 "Sequential learning: học từ lỗi của mô hình trước. AdaBoost, GradientBoostingClassifier. Learning rate và n_estimators.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 4: XGBoost - Extreme Gradient Boosting",
                 "Thuật toán 'vua' của competitions. Hyperparameters quan trọng: max_depth, learning_rate, subsample. Early stopping.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 5: LightGBM & CatBoost",
                 "Alternatives cho XGBoost. LightGBM: tốc độ cao. CatBoost: xử lý categorical tự động. So sánh performance.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Stacking & Blending",
                 "Meta-learning: train mô hình trên predictions của các mô hình khác. StackingClassifier trong scikit-learn.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Advanced Project
        card("Competition Project: Kaggle-Style Challenge",
             "Dataset thi đấu: feature engineering nâng cao, ensemble nhiều mô hình (RF + XGBoost + LightGBM), stacking, hyperparameter tuning, leaderboard simulation.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 5: MODEL DEPLOYMENT ==================
    st.markdown("---")
    st.header("🚀 Chương 5: Model Deployment & MLOps")
    st.info("🎯 Mục tiêu: Triển khai mô hình thực tế, monitor performance, versioning và CI/CD cho ML")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Model Serialization",
                 "Lưu mô hình với pickle, joblib. Model versioning. Lưu preprocessing pipeline cùng với model.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Web API với FastAPI/Flask",
                 "Tạo REST API cho model prediction. JSON input/output. Error handling và validation. Docker containerization.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Streamlit Web App",
                 "Tạo demo app nhanh chóng. File upload, real-time prediction, visualization. Deploy lên Streamlit Cloud.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 4: Model Monitoring",
                 "Data drift, model drift detection. Performance monitoring in production. Alerting system.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 5: A/B Testing cho ML",
                 "Thử nghiệm mô hình mới vs cũ. Statistical significance. Multi-armed bandit approach.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: CI/CD Pipeline",
                 "Automated testing cho ML code. Model validation pipeline. GitHub Actions cho ML deployment.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Final Project
        card("Capstone Project: End-to-End ML System",
             "Xây dựng hệ thống ML hoàn chỉnh: từ data pipeline, feature engineering, model training, API deployment, monitoring dashboard, A/B testing setup.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    st.markdown("---")
    st.success("👉 Hoàn thành Machine Learning track, bạn sẽ có kiến thức vững về các thuật toán ML, biết deploy model production và xây dựng end-to-end ML system!")