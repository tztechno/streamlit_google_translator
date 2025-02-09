import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import base64
import io
import os
import zipfile
import gdown

# Page config
st.set_page_config(page_title="Japanese to English Translator", layout="wide")

# シンプルな音声再生用のHTML関数
def create_audio_player(audio_data):
    b64 = base64.b64encode(audio_data).decode()
    md = f"""
        <audio autoplay controls style="width: 100%">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return st.markdown(md, unsafe_allow_html=True)

# Google Drive からダウンロード
def download_and_extract_model():
    model_dir = "./model"
    zip_path = "./model.zip"
    file_id = "1ZvD0-kH_Yt9GuMmf0KRcsNLJHOHsQEAF"  # ← ここを変更（Google Drive のファイル ID）

    if not os.path.exists(model_dir):  # 既にモデルがあるならスキップ
        st.info("Downloading model... (This may take a while)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

        st.info("Extracting model...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./")

        os.remove(zip_path)  # ZIP 削除

# モデルのロード
@st.cache_resource
def load_model():
    download_and_extract_model()  # 必要ならダウンロード
    model_path = "./model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# 翻訳関数
def translate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        return tokenizer.decode(output[0], skip_special_tokens=True)

# 音声生成
def generate_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.error(f"音声生成エラー: {str(e)}")
        return None

# Session state 初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI
st.title("Japanese to English Translator")

# サイドバー設定
with st.sidebar:
    enable_tts = st.checkbox("英語の音声を再生", value=True)

# チャット履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 翻訳入力
if prompt := st.chat_input("翻訳したい日本語を入力してください"):
    # ユーザー入力を表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 翻訳結果を生成
    with st.chat_message("assistant"):
        translation = translate_text(prompt)
        st.markdown(translation)
        st.session_state.messages.append({"role": "assistant", "content": translation})
        
        # 音声再生
        if enable_tts:
            with st.spinner("音声生成中..."):
                audio_data = generate_speech(translation)
                if audio_data:
                    create_audio_player(audio_data)
