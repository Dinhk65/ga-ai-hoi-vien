import streamlit as st
from video_card import video_card
from card import card

# ================== PAGE: DEEP LEARNING ==================
def deep_learning_page():
    st.title("🧠 Deep Learning – Từ Neural Network đến AI hiện đại")
    st.info("Từ perceptron đơn giản đến Transformer - nền tảng của ChatGPT và các AI model tiên tiến")

    # ================== CHƯƠNG 1 ==================
    st.markdown("---")
    st.header("🔥 Chương 1: Neural Networks Cơ Bản")
    st.info("🎯 Mục tiêu: Hiểu bản chất neural network, từ perceptron đến MLP, làm chủ backpropagation")

    with st.expander("Xem chi tiết: "):
        # Bài học
        col1, _, col2 = st.columns([8, 1, 8])
        with col1:
            card("Bài 1: Từ não bộ đến AI",
                 "Neurons sinh học vs artificial neurons. Perceptron đơn giản: inputs, weights, bias, activation. Tại sao cần nhiều layers?",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Multi-layer Perceptron (MLP)",
                 "Architecture: Input → Hidden → Output layers. Forward pass step by step. Activation functions: Sigmoid, ReLU, Tanh, Softmax.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Backpropagation - trực quan siêu dễ hiểu",
                 "Thuật toán 'học' của neural network. Chain rule in action. Gradient flow qua các layers. Visualize với computational graph.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Thực hành với TensorFlow/Keras",
                 "Xây dựng MLP đầu tiên: Sequential model, Dense layers, compile, fit. Hyperparameters: learning_rate, batch_size, epochs.",
                 icon="📑",
                 color='#e8f5e9'
                 )
        with col2:
            card("Bài 5: Loss Functions cho Deep Learning",
                 "Binary crossentropy, Categorical crossentropy, Sparse categorical crossentropy. MSE cho regression. Custom loss functions.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Optimizers nâng cao",
                 "SGD, Adam, RMSprop, AdaGrad. Learning rate scheduling. Adaptive learning rates. So sánh convergence speed.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Regularization trong Deep Learning",
                 "Overfitting trong neural networks. Dropout, Batch Normalization, L1/L2 regularization. Early stopping với callbacks.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Debugging Neural Networks",
                 "Learning curves, gradient vanishing/exploding. Weight initialization. Monitoring training process với TensorBoard.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        card("Mini Project 1: Phân loại chữ số viết tay",
             "Dataset MNIST: xây dựng MLP từ scratch, so sánh activation functions, visualize learned weights, achieve >95% accuracy.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Mini Project 2: Dự đoán giá nhà với Neural Network",
             "Dataset Boston Housing: MLP cho regression, feature scaling, hyperparameter tuning, so sánh với linear regression.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 2 ==================
    st.markdown("---")
    st.header("🖼️ Chương 2: Convolutional Neural Networks (CNN)")
    st.info("🎯 Mục tiêu: Làm chủ CNN cho Computer Vision - từ edge detection đến image classification nâng cao")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Tại sao MLP không phù hợp với ảnh?",
                 "Vấn đề với high-dimensional images. Spatial structure và translation invariance. Convolution operation trực quan.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Convolution & Feature Maps",
                 "Kernels/Filters học detect patterns. Stride, padding, feature maps. Visualize convolution step-by-step với animations.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Pooling & CNN Architecture",
                 "Max pooling, Average pooling. CNN = Conv → Pool → Conv → Pool → Flatten → Dense. Receptive field concept.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: CNN với Keras",
                 "Conv2D, MaxPool2D, Flatten layers. Input shape cho images. Data augmentation với ImageDataGenerator. Model summary visualization.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: Advanced CNN Techniques",
                 "Batch Normalization trong CNN. Dropout cho overfitting. Global Average Pooling thay cho Dense layers cuối.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Transfer Learning",
                 "Sử dụng pre-trained models: VGG16, ResNet50, MobileNet. Feature extraction vs Fine-tuning. Freezing layers.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: CNN Architectures nổi tiếng",
                 "LeNet, AlexNet, VGG, ResNet, Inception. Residual connections giải quyết vanishing gradient. Architecture evolution.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Object Detection cơ bản",
                 "Image classification vs Object detection vs Segmentation. YOLO concept. Bounding boxes và confidence scores.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Projects
        card("Project 1: Image Classification",
             "Dataset CIFAR-10: xây dựng CNN từ đầu, data augmentation, visualize feature maps, achieve >80% accuracy, so sánh với pre-trained model.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: Medical Image Analysis",
             "Dataset X-ray pneumonia: transfer learning với ResNet50, fine-tuning, class imbalance handling, medical AI ethics considerations.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 3 ==================
    st.markdown("---")
    st.header("🔄 Chương 3: Recurrent Neural Networks (RNN)")
    st.info("🎯 Mục tiêu: Xử lý dữ liệu sequence với RNN, LSTM, GRU - ứng dụng NLP và time series")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Sequential Data & RNN Motivation",
                 "Dữ liệu chuỗi: text, time series, speech. Tại sao CNN/MLP không phù hợp? RNN architecture với hidden state memory.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Vanilla RNN",
                 "RNN cell: input + previous hidden state → new hidden state. Many-to-one, one-to-many, many-to-many architectures. Vanishing gradient problem.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Long Short-Term Memory (LSTM)",
                 "Giải quyết vanishing gradient. Forget gate, Input gate, Output gate. Cell state vs Hidden state. LSTM intuition với analogies.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Gated Recurrent Unit (GRU)",
                 "LSTM đơn giản hóa. Reset gate, Update gate. So sánh LSTM vs GRU: performance, computational cost.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: RNN với Keras",
                 "SimpleRNN, LSTM, GRU layers. return_sequences parameter. Bidirectional RNNs. Stacking RNN layers.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Text Preprocessing cho RNN",
                 "Tokenization, Vocabulary building. Padding sequences. Word embeddings: Word2Vec concept, Embedding layer trong Keras.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Sequence-to-Sequence Models",
                 "Encoder-Decoder architecture. Language translation intuition. Attention mechanism preview (chuẩn bị cho Transformer).",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Time Series Forecasting",
                 "RNN cho dự báo. Sliding window approach. Multi-step prediction. Evaluation metrics: MAE, RMSE cho time series.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Projects
        card("Project 1: Sentiment Analysis",
             "Dataset movie reviews: text preprocessing, word embeddings, LSTM classifier, compare với traditional ML, interpretability với attention weights.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: Stock Price Prediction",
             "Dataset historical stock prices: feature engineering, LSTM time series model, walk-forward validation, risk disclaimer và limitations.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 4 ==================
    st.markdown("---")
    st.header("⚡ Chương 4: Transformer Architecture")
    st.info("🎯 Mục tiêu: Hiểu kiến trúc Transformer - nền tảng của ChatGPT, BERT, GPT và các LLM hiện đại")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Attention is All You Need",
                 "Vấn đề của RNN: sequential processing, long dependencies. Self-attention mechanism. Query, Key, Value concept trực quan.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Self-Attention Mechanism",
                 "Scaled Dot-Product Attention step-by-step. Attention weights visualization. Multi-Head Attention: nhiều 'góc nhìn' khác nhau.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Transformer Block Architecture",
                 "Encoder stack: Multi-Head Attention + Feed-Forward + Residual connections + Layer Normalization. Positional Encoding cho sequence order.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Decoder & Language Modeling",
                 "Decoder với Masked Self-Attention. Cross-Attention giữa Encoder-Decoder. Autoregressive generation process.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: BERT - Bidirectional Understanding",
                 "Pre-training với Masked Language Modeling. [CLS], [SEP] tokens. Fine-tuning cho downstream tasks: classification, NER, QA.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: GPT - Generative Pre-training",
                 "Decoder-only architecture. Next token prediction. Zero-shot, Few-shot learning. GPT-1 → GPT-2 → GPT-3 → ChatGPT evolution.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Vision Transformer (ViT)",
                 "Transformer cho Computer Vision. Image patches như 'words'. Position embeddings cho 2D images. ViT vs CNN comparison.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Scaling Laws & Large Language Models",
                 "Model size, data size, compute scaling. Emergent abilities trong large models. Alignment, RLHF. Responsible AI considerations.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Advanced Projects
        card("Project 1: Text Summarization",
             "Dataset news articles: implement Transformer Encoder-Decoder, ROUGE evaluation, compare với extractive methods, attention visualization.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: Question Answering System",
             "Dataset SQuAD-style: fine-tune pre-trained BERT, implement extractive QA, evaluation metrics, deploy với HuggingFace pipeline.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 5 ==================
    st.markdown("---")
    st.header("🚀 Chương 5: Advanced Deep Learning & Applications")
    st.info("🎯 Mục tiêu: Các kỹ thuật DL nâng cao, GANs, deployment và xu hướng AI mới nhất")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Generative Adversarial Networks (GANs)",
                 "Generator vs Discriminator game theory. Vanilla GAN training process. Mode collapse, training instability. Applications: image generation, data augmentation.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Advanced GAN Architectures",
                 "DCGAN cho stable training. StyleGAN cho high-quality faces. Conditional GANs. CycleGAN cho style transfer. Ethical considerations.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Autoencoders & Variational AE",
                 "Encoder-Decoder cho unsupervised learning. Latent space representation. VAE cho probabilistic generation. Applications: anomaly detection, denoising.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 4: Model Optimization & Deployment",
                 "Model quantization, pruning, distillation. TensorFlow Lite, ONNX. Edge deployment considerations. Inference optimization techniques.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 5: MLOps cho Deep Learning",
                 "Experiment tracking với Weights & Biases. Model versioning. GPU training workflows. Docker containers cho reproducibility.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Emerging Trends & Future",
                 "Multimodal models (CLIP, DALL-E). Reinforcement Learning từ Human Feedback. Foundation models. AI Safety và Alignment challenges.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Capstone Project
        card("Capstone Project: End-to-End AI Application",
             "Multimodal project: CNN cho image processing + Transformer cho text analysis + web deployment. Ví dụ: AI-powered content moderation system hoặc medical diagnosis assistant.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        # Portfolio Projects
        card("Portfolio Showcase: AI Demos",
             "Tạo collection các demo AI: chatbot với Transformer, image generator với GAN, style transfer app, object detection system. Deploy lên cloud platforms.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    st.markdown("---")
    st.success("👉 Hoàn thành Deep Learning track, bạn sẽ hiểu sâu về neural networks, làm chủ CNN, RNN, Transformer và có thể xây dựng các ứng dụng AI hiện đại!")