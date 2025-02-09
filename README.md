# streamlit_je_translator

---

import os
import zipfile
import gdown
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Google Drive からダウンロード
def download_and_extract_model():
    model_dir = "./model"
    zip_path = "./model.zip"
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"  # ← ここを変更（Google Drive のファイル ID）

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
    
---
