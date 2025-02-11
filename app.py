
import streamlit as st
from deep_translator import GoogleTranslator
import tempfile
import os
from gtts import gTTS

def main():
    st.title("Multi Translation")
    
    # Get list of available languages
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
    
    # Create input text area
    input_text = st.text_area("Enter text to translate:", value="Hello")
    
    # Create language selection dropdown
    target_lang = st.selectbox(
        "Select target language:",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]
    )
    
    if st.button("Translate and Generate Audio"):
        if input_text:
            try:
                # Translate text
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated_text = translator.translate(input_text)
                
                # Display translated text
                # st.write("### Translated Text:")
                st.write(translated_text)
                
                # Generate audio file
                tts = gTTS(text=translated_text, lang=target_lang)
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_file = fp.name
                    tts.save(temp_file)
                
                # Display audio player
                # st.write("### Audio:")
                st.audio(temp_file)
                
                # Clean up temporary file
                os.unlink(temp_file)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main()

