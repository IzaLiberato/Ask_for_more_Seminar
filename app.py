from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from transformers import pipeline, T5TokenizerFast, DistilBertTokenizer, DistilBertForQuestionAnswering
import fitz
import pytesseract
from pdf2image import convert_from_path
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#testar se a extensão é permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            text += page.get_text()
        return text
    except:
        images = convert_from_path(pdf_path)
        text = ''
        for image in images:
            text += pytesseract.image_to_string(image)
        return text

max_model_length = 512
tokenizer_t5 = T5TokenizerFast.from_pretrained("t5-base", model_max_length=max_model_length)

#resumi o texto
def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-base", tokenizer=tokenizer_t5)

    #dividi o txt
    segments = [text[i:i + 512] for i in range(0, len(text), 512)]

    #ajusta max_length com base no comprimento real dos segmentos
    max_lengths = [min(len(segment), 150) for segment in segments]

    #concatena segmentos
    summaries = [summarizer(segment, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text'] for segment, max_length in zip(segments, max_lengths)]

    return ' '.join(summaries)

# Carregar modelo de Question Answering
tokenizer_qa = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model_qa = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Função para responder perguntas sobre o artigo
def answer_questions(article_text, question):
    inputs = tokenizer_qa(question, article_text, return_tensors="pt")
    start_positions = model_qa(**inputs).start_logits.argmax(dim=1)
    end_positions = model_qa(**inputs).end_logits.argmax(dim=1)
    answer = tokenizer_qa.convert_tokens_to_string(tokenizer_qa.convert_ids_to_tokens(inputs["input_ids"][0][start_positions[0]:end_positions[0]+1]))
    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Nenhum arquivo enviado.')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='Nome de arquivo inválido.')

        if not allowed_file(file.filename):
            return render_template('index.html', error='Extensão de arquivo não permitida. Por favor, envie um arquivo PDF.')

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Extrai texto do PDF
        pdf_text = extract_text_from_pdf(f'uploads/{filename}')

        # Resumir o texto
        summarized_text = summarize_text(pdf_text)

        # Receber pergunta do usuário
        user_question = request.form['question']

        # Responder à pergunta usando o modelo de Question Answering
        answer = answer_questions(pdf_text, user_question)

        return render_template('index.html', filename=filename, pdf_text=pdf_text, summarized_text=summarized_text, user_question=user_question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
