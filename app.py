# app.py
import subprocess
import uuid
from flask import Flask, request, jsonify, send_file
import requests
from werkzeug.utils import secure_filename
import os
import ffmpeg
from scipy.spatial import distance


def create_app():
    app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
    app.config['UPLOAD_FOLDER'] = '/app/uploads/'
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    # Other setup code...
    return app


app = create_app()


@app.route('/', methods=['GET'])
def homepage():
    return "Homepage"


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello"

@app.route('/get_similar', methods=['POST'])
def cosine_similarity():
    data = request.json
    query_vector = data['query_vector']
    vector_text_pairs = data['vectors']

    # Extract embeddings and their corresponding texts
    vectors = [pair['embeddings'] for pair in vector_text_pairs]
    texts = [pair['text'] for pair in vector_text_pairs]

    # Calculate cosine similarity for each vector
    # Return the index of the most similar vector
    most_similar_index = max(range(len(vectors)), key=lambda index: 1 - distance.cosine(query_vector, vectors[index]))

    return jsonify({'most_similar_text': texts[most_similar_index]})


from flask import Flask, request, render_template, send_file
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name)

# Load a pre-trained question-answering model
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input and uploaded document
        user_question = request.form["question"]
        uploaded_document = request.files["document"]

        # Process the uploaded document
        document_text = process_uploaded_document(uploaded_document)

        # Answer the user's question based on the document
        answer = answer_question(user_question, document_text)

        # Create a PDF with the answer
        create_pdf(answer, "response.pdf")

        return render_template("home.html", answer=answer)

    return render_template("home.html", answer=None)

def process_uploaded_document(uploaded_document):
    # Use a document processing library to extract and preprocess the document text
    # Return the document text
    pass

def answer_question(question, document_text):
    # Tokenize the question and document
    inputs = tokenizer(question, document_text, return_tensors="pt", padding=True, truncation=True)
    
    # Get the answer
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

def create_pdf(answer, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Answer:")
    c.drawString(100, 725, answer)
    c.save()

    return filename

@app.route('/download-pdf')
def download_pdf():
    return send_file("response.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
