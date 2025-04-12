import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import easyocr

st.set_page_config(layout="wide")

# تحميل قارئ النصوص
@st.cache_resource
def load_reader():
    return easyocr.Reader(['ar', 'en'])
reader = load_reader()

# دالة استخراج خصائص بسيطة من الصورة (لون متوسط)
def extract_simple_features(img):
    img = img.resize((100, 100)).convert("RGB")
    arr = np.array(img).reshape(-1, 3)
    return np.mean(arr, axis=0)

# دالة تصنيف النصوص
def classify_text(text):
    text_lower = text.lower()
    if any(k in text_lower for k in ['بشرة', 'زيت', 'عناية', 'ترطيب']):
        return 'نصائح للعناية بالبشرة'
    elif any(k in text_lower for k in ['nlp', 'ai', 'ml', 'بودكاست']):
        return 'تعليم / تقنية'
    elif any(k in text_lower for k in ['netflix', 'مسلسل']):
        return 'محتوى ترفيهي'
    elif any(k in text_lower for k in ['مرحبا', 'تدريب', 'خدمة']):
        return 'محادثة مهمة'
    elif any(k in text_lower for k in ['وتشاء', 'رحمه الله', 'اقتباس']):
        return 'اقتباسات / خواطر'
    else:
        return 'غير مصنفة'

st.set_page_config(layout="wide")
st.title("مقبرة لقطات الشاشة الذكية *･ﾟ✧")

uploaded_files = st.file_uploader("ارفع لقطات الشاشة", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    features = []
    data = []

    for file in uploaded_files:
        img = Image.open(file)
        feat = extract_simple_features(img)

        # استخدام easyocr لاستخراج النصوص
        result = reader.readtext(np.array(img))
        text = "\n".join([item[1] for item in result])

        category = classify_text(text)
        features.append(feat)
        data.append({
            "file_name": file.name,
            "image": img,
            "text": text.strip(),
            "category": category
        })

    # تجميع بصري باستخدام KMeans
    k = st.slider("كم مجموعة تقريبًا تبين؟", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)

    for i, label in enumerate(labels):
        data[i]["group"] = label

    df = pd.DataFrame(data)

    # حقل بحث
    query = st.text_input("ابحث داخل لقطات الشاشة:")
    if query:
        df = df[df["text"].str.contains(query, case=False, na=False)]

    # عرض المجموعات
    grouped = df.groupby("group")
    for group, group_df in grouped:
        st.subheader(f"المجموعة {group + 1} *")
        cols = st.columns(3)
        for i, row in group_df.iterrows():
            with cols[i % 3]:
                st.image(row["image"], caption=row["file_name"], use_column_width=True)
                st.markdown(f"**التصنيف:** {row['category']}")
                st.text_area("النص:", row["text"], height=150)

    # تحميل النتائج
    result_df = df[["file_name", "category", "text", "group"]]
    st.download_button("تحميل النتائج كـ Excel", result_df.to_csv(index=False).encode("utf-8"), file_name="لقطات_الشاشة.csv", mime="text/csv")
