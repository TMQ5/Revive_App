import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
import pytesseract

# تحميل النموذج
@st.cache_resource
def load_model():
    return MobileNet(weights='imagenet', include_top=False, pooling='avg')
model = load_model()

def extract_features(img):
    img = img.resize((224, 224)).convert('RGB')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]

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
        feat = extract_features(img)
        text = pytesseract.image_to_string(img, lang='ara+eng')
        category = classify_text(text)
        features.append(feat)
        data.append({
            "file_name": file.name,
            "image": img,
            "text": text.strip(),
            "category": category
        })

    # تجميع الصور حسب التشابه البصري
    k = st.slider("كم مجموعة تقريبًا تبين؟", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)

    # إضافة الملصقات للمجموعات
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
