import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load model
@st.cache_resource
def load_model():
    save_dir = "./fake_news_phobert"   # thư mục đã lưu model
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    id2label = {0: "REAL", 1: "FAKE"}
    model.config.id2label = id2label
    model.config.label2id = {v: k for k, v in id2label.items()}

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU; nếu deploy có GPU thì để 0
    )
    return clf

st.title("📰 Fake News Detector (PhoBERT)")

clf = load_model()

# Input text
text = st.text_area("Nhập văn bản cần kiểm tra:", height=150)

if st.button("Dự đoán"):
    if text.strip():
        result = clf(text)[0]
        st.write(f"**Dự đoán:** {result['label']}")
        st.write(f"**Độ tin cậy:** {result['score']:.2f}")
    else:
        st.warning("Hãy nhập văn bản trước khi dự đoán.")