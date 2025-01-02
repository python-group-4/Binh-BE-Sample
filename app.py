from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import re

app = Flask(__name__)

# Cấu hình API của Google Gemini
genai.configure(api_key="AIzaSyBh7k24t58qJ-cP2AIhmQQdqVJ4MtsbVeU")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# ==============================
# HÀM HỖ TRỢ
# ==============================

def get_youtube_transcript(url):
    """Lấy phụ đề từ URL YouTube."""
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if not video_id_match:
        return None, "URL không hợp lệ."

    video_id = video_id_match.group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text, None
    except Exception:
        return None, "Không thể tìm thấy phụ đề cho video này."

def query_gemini(prompt):
    """Gửi prompt tới mô hình Gemini."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error querying Gemini:", e)
        return "Không thể thực hiện tóm tắt tại thời điểm này."

def generate_mindmap(text):
    """Tạo mindmap từ bản tóm tắt văn bản dưới dạng JSON."""
    if not text.strip():
        return {"nodes": []}  # Trả về mindmap trống nếu văn bản rỗng

    sentences = text.split('. ')  # Tách văn bản thành từng câu
    mindmap = {"nodes": []}

    for idx, sentence in enumerate(sentences):
        # Phân loại các ý chính và chi tiết
        if len(sentence.split()) > 5:  # Giả sử câu có hơn 5 từ là ý chính
            node = {
                "id": f"main_{idx}",
                "text": sentence[:100],  # Giới hạn 100 ký tự
                "type": "main",
                "children": []
            }
            mindmap["nodes"].append(node)
        else:
            node = {
                "id": f"detail_{idx}",
                "text": sentence[:100],  # Giới hạn 100 ký tự
                "type": "detail"
            }
            # Thêm câu chi tiết vào node trước đó nếu có
            if mindmap["nodes"]:
                mindmap["nodes"][-1]["children"].append(node)

    return mindmap

# ==============================
# API ROUTES
# ==============================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return jsonify({"message": "Trang chủ đang hoạt động!"})
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    url = data.get('url', '')

    if not url:
        return jsonify({"error": "Không có URL được cung cấp."}), 400

    # Lấy phụ đề từ video YouTube
    transcript, error = get_youtube_transcript(url)
    if error:
        return jsonify({"error": error}), 400

    # Tóm tắt văn bản bằng Gemini
    summary_prompt = f"Tóm tắt văn bản sau:\n{transcript}"
    summary = query_gemini(summary_prompt)

    # Tạo mindmap từ bản tóm tắt
    mindmap = generate_mindmap(summary)

    return jsonify({
        "summary": summary,
        "mindmap": mindmap
    })

if __name__ == "__main__":
    app.run(debug=True)
