import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
from datetime import timedelta

st.set_page_config(page_title="Bablu Transcriber", layout="centered")
st.title("🧠 Bablu Transcriber")

st.markdown("### 🎥 Upload your Video file (MP4)")
video_file = st.file_uploader("Choose a file", type=["mp4", "mkv", "mov", "webm"])

language = st.selectbox("Source Language (Auto-detect recommended)", ["auto", "en", "hi", "ta", "te", "ur"])
output_lang = st.selectbox("Translate to", ["Hindi"])
output_format = st.selectbox("Output Format", ["TXT", "SRT", "Excel"])
submit = st.button("🎬 Start Transcription")

if submit and video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        temp_path = tmp_file.name

    st.info("⏳ Loading Whisper model...")
    model = whisper.load_model("base")

    st.info("🎧 Transcribing and translating... this might take some time.")

    result = model.transcribe(temp_path, task="translate", language=None if language == "auto" else language)

    # Build output data
    lines = []
    srt_lines = []
    excel_rows = []

    for i, seg in enumerate(result["segments"], start=1):
        start = str(timedelta(seconds=int(seg["start"])))
        end = str(timedelta(seconds=int(seg["end"])))
        text = seg["text"].strip()

        # TXT Line
        lines.append(f"[{start} --> {end}] {text}")

        # SRT Line
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

        # Excel row
        excel_rows.append({
            "Start": start,
            "End": end,
            "Dialogue": text
        })

    if output_format == "TXT":
        txt_output = "\n".join(lines)
        st.download_button("📄 Download TXT", txt_output, file_name="transcription.txt")

    elif output_format == "SRT":
        srt_output = "\n".join(srt_lines)
        st.download_button("🎬 Download SRT", srt_output, file_name="subtitles.srt")

    elif output_format == "Excel":
        df = pd.DataFrame(excel_rows)
        excel_bytes = df.to_csv(index=False, encoding='utf-8')
        st.download_button("📊 Download Excel", excel_bytes, file_name="transcription.csv")

    st.success("✅ Transcription complete!")
