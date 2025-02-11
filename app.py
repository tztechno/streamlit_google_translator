import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import io

# --- 事前にクリックを要求（スマホ対応） ---
if "user_clicked" not in st.session_state:
    st.session_state.user_clicked = False

if not st.session_state.user_clicked:
    if st.button("🔊 Enable Audio Autoplay"):
        st.session_state.user_clicked = True
    st.stop()  # ボタンが押されるまで処理を止める

def create_audio_player(audio_data, autoplay=True):
    """音声を再生するHTMLプレイヤー"""
    b64 = base64.b64encode(audio_data).decode()
    
    # autoplay を有効化（ユーザーがクリック済みなら動作可能）
    auto_attr = "autoplay" if autoplay else ""
    
    md = f"""
        <audio {auto_attr} controls style="width: 100%">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return st.markdown(md, unsafe_allow_html=True)

def generate_speech(text, lang):
    """翻訳されたテキストを音声に変換"""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    st.title("Multi Translation")
    
    LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'ja': 'Japanese',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ko': 'Korean',        
        'zh-CN':'Chinese (simplified)',
        'ar': 'Arabic',
    }
    
    input_text = st.text_area("Enter text to translate:", value="Hello")
    target_lang = st.selectbox("Select target language:", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

    if st.button("Translate and Generate Audio"):
        if input_text:
            try:
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated_text = translator.translate(input_text)
                st.write(translated_text)
                
                # 音声生成
                audio_data = generate_speech(translated_text, target_lang)
                
                if audio_data:
                    create_audio_player(audio_data, autoplay=True)  # 自動再生（ユーザー操作後なら可能）

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main()
