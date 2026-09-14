import os
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from disease_info import get_info

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Plant Disease Detector 🌱",
    page_icon="🌱",
    layout="wide",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "plant_disease_final_96_89.keras")

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# =========================================================
# MODEL LOADING (cached so it only loads once per session)
# =========================================================
@st.cache_resource(show_spinner="Loading model... (first run only)")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image: Image.Image, model):
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    predicted_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[predicted_index] * 100)

    top_indices = np.argsort(preds)[-5:][::-1]
    top5 = [
        {"class": CLASS_NAMES[int(i)], "confidence": round(float(preds[i] * 100), 2)}
        for i in top_indices
    ]
    return predicted_class, confidence, top5


# =========================================================
# CHATBOT LOGIC
# =========================================================
def get_groq_client():
    """Return a Groq client if an API key is configured, else None."""
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


def build_system_prompt(predicted_class):
    if predicted_class:
        info = get_info(predicted_class)
        context = (
            f"The user just scanned a plant leaf and the model predicted: {info['name']}.\n"
            f"Symptoms: {info['symptoms']}\n"
            f"Cause: {info['cause']}\n"
            f"Treatment: {info['treatment']}\n"
            f"Prevention: {info['prevention']}\n"
        )
    else:
        context = "The user has not scanned any leaf image yet in this session."

    return (
        "You are a friendly, knowledgeable plant-health assistant embedded in a "
        "plant disease detection app. Answer clearly and concisely, give practical, "
        "safe agricultural advice, and mention when the user should consult a local "
        "agronomist/expert for serious or uncertain cases.\n\n" + context
    )


def rule_based_reply(user_msg, predicted_class):
    """Fallback answer used when no LLM API key is configured."""
    msg = user_msg.lower()

    if not predicted_class:
        return (
            "Please upload and scan a leaf image first from the left panel — "
            "then ask me things like 'what is the treatment?' or 'how do I prevent this?'"
        )

    info = get_info(predicted_class)

    if any(w in msg for w in ["treat", "cure", "medicine", "fungicide", "spray"]):
        return f"**Treatment for {info['name']}:**\n\n{info['treatment']}"
    if any(w in msg for w in ["symptom", "sign", "look like"]):
        return f"**Symptoms of {info['name']}:**\n\n{info['symptoms']}"
    if any(w in msg for w in ["cause", "why", "reason"]):
        return f"**Cause of {info['name']}:**\n\n{info['cause']}"
    if any(w in msg for w in ["prevent", "avoid", "stop it from"]):
        return f"**Prevention for {info['name']}:**\n\n{info['prevention']}"

    return (
        f"**{info['name']}**\n\n"
        f"- Symptoms: {info['symptoms']}\n"
        f"- Cause: {info['cause']}\n"
        f"- Treatment: {info['treatment']}\n"
        f"- Prevention: {info['prevention']}\n\n"
        "Ask me specifically about symptoms, cause, treatment, or prevention for more detail."
    )


def get_bot_response(user_msg, predicted_class, chat_history):
    client = get_groq_client()

    if client is None:
        return rule_based_reply(user_msg, predicted_class)

    try:
        messages = [{"role": "system", "content": build_system_prompt(predicted_class)}]
        for m in chat_history[-8:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.4,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Chat API error, using offline mode) \n\n{rule_based_reply(user_msg, predicted_class)}"


# =========================================================
# SESSION STATE
# =========================================================
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "top5" not in st.session_state:
    st.session_state.top5 = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# HEADER
# =========================================================
st.title("🌱 Plant Disease Detection & Assistant")
st.caption("Upload a leaf image to detect the disease (38 classes), then chat with the assistant about it.")

col_left, col_right = st.columns([1, 1.2])

# =========================================================
# LEFT: IMAGE UPLOAD + PREDICTION
# =========================================================
with col_left:
    st.subheader("📷 Scan a Leaf")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Detect Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf..."):
                model = load_model()
                predicted_class, confidence, top5 = predict_image(image, model)
                st.session_state.predicted_class = predicted_class
                st.session_state.confidence = confidence
                st.session_state.top5 = top5

    if st.session_state.predicted_class:
        info = get_info(st.session_state.predicted_class)
        st.success(f"**Prediction:** {info['name']} ({st.session_state.confidence:.2f}% confidence)")

        with st.expander("Top 5 predictions"):
            for item in st.session_state.top5:
                nm = get_info(item["class"])["name"]
                st.write(f"- {nm}: {item['confidence']}%")

        with st.expander("ℹ️ Disease info", expanded=True):
            st.markdown(f"**Symptoms:** {info['symptoms']}")
            st.markdown(f"**Cause:** {info['cause']}")
            st.markdown(f"**Treatment:** {info['treatment']}")
            st.markdown(f"**Prevention:** {info['prevention']}")

# =========================================================
# RIGHT: CHATBOT
# =========================================================
with col_right:
    st.subheader("💬 Ask the Plant Assistant")

    if get_groq_client() is None:
        st.info(
            "Running in **offline mode** (no GROQ_API_KEY set) — the assistant will "
            "answer using the built-in disease knowledge base. Add a free Groq API "
            "key in `.streamlit/secrets.toml` for smarter, open-ended chat.",
            icon="ℹ️",
        )

    chat_container = st.container(height=420)
    with chat_container:
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    user_msg = st.chat_input("Ask about symptoms, treatment, prevention...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        reply = get_bot_response(user_msg, st.session_state.predicted_class, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
