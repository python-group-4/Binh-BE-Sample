from flask import Flask, render_template, request, jsonify
import aiohttp
import asyncio
import json
import google.generativeai as genai
import time
from transformers import GPT2Tokenizer
import re
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

# API URL của Hugging Face (thay bằng API của bạn)
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {
    "Authorization": "Bearer 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"
}


# Hàm lấy phụ đề từ URL YouTube
def get_youtube_transcript(url):
    video_id = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if not video_id:
        return "URL không hợp lệ"
    video_id = video_id.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        return "Không thể tìm thấy phụ đề cho video này."
    
    
# Hàm gọi API Hugging Face để tóm tắt văn bản bất đồng bộ
async def fetch_summary(session, chunk):
    async with session.post(HF_API_URL, headers=HEADERS, json={"inputs": chunk}) as response:
        if response.status == 200:
            result = await response.json()
            return result[0]['summary_text']
        else:
            return "Error in summarization."

# Hàm tóm tắt văn bản bất đồng bộ
async def summarize_text_async(text):
    max_token_length = 1024  # Giới hạn token cho BART model, có thể thay đổi tùy theo mô hình
    text_chunks = [text[i:i + max_token_length] for i in range(0, len(text), max_token_length)]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_summary(session, chunk) for chunk in text_chunks]
        summaries = await asyncio.gather(*tasks)
    
    return " ".join(summaries)

# Hàm tạo mind map dưới dạng JSON từ văn bản
def generate_mindmap(text):
    sentences = text.split('. ')
    mindmap = {"nodes": []}
    
    for idx, sentence in enumerate(sentences):
        node = {
            "id": idx,
            "text": sentence[:100]  # Chỉ lấy 100 ký tự đầu của mỗi câu
        }
        mindmap["nodes"].append(node)
        
    return mindmap

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["text_input"]
        # Sử dụng async để tóm tắt văn bản
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary = loop.run_until_complete(summarize_text_async(input_text))
        mindmap = generate_mindmap(summary)
        return jsonify({
            "summary": summary,
            "mindmap": mindmap
        })
    return render_template("index.html")


# Function to tokenize and split text
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
def tokenize_and_split(text, max_tokens):
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    # Split into chunks
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]

    # Decode tokens back into text for each chunk
    chunk_texts = [tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in chunks]

    return chunk_texts

# Function to query Gemini API
# genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
def query_gemini(prompt):   
    response = model.generate_content(prompt)
    return response.text


# API tóm tắt văn bản từ URL YouTube và tạo mindmap
@app.route('/summarize', methods=['POST'])
def summarize():
    max_tokens_per_chunk = 10000
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Lấy văn bản phụ đề từ video YouTube
    text = get_youtube_transcript(url)
    if text == "Không thể tìm thấy phụ đề cho video này.":
        return jsonify({"error": text}), 400

    # Chia văn bản thành các đoạn nhỏ hơn nếu văn bản quá dài
    chunks = tokenize_and_split(text, max_tokens_per_chunk)

    summary_texts = ""
    for idx, chunk in enumerate(chunks):
        print(f"Querying Gemini with chunk {idx + 1}...")
        response = query_gemini(f'This is a paragrapth: {chunk} of a big video transcript. Please summerize it. content of the summerize: ')
        summary_texts += response
    # Sleep to resest free gemini token
    if idx > 2:
        time.sleep(5)
        
    time.sleep(10)
    grand_summary = query_gemini(f"""
                # Task
                You are given multiple summary parts from a videos. Try to summerize them into a big picture summarize that can cover all the important information
                # Output
                The output should be in well markdown format 
                # List of summary
                {summary_texts}
                # The summary: 
            """)
    
    return jsonify({
        "summary": grand_summary.strip(),
    })
        
if __name__ == "__main__":
    app.run(debug=True)
