from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from rouge_score import rouge_scorer
from pydantic import BaseModel
from pythainlp import word_tokenize
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load semantic model
semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@app.get("/")
def read_root():
    return FileResponse("index.html")

class ThaiTokenizer:
    def tokenize(self, text: str):
        return word_tokenize(text, engine='newmm', keep_whitespace=False)

class TextInput(BaseModel):
    reference: str
    candidate: str

@app.post("/score")
def calculate_rouge(input: TextInput):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rougeL'],
        tokenizer=ThaiTokenizer(),
        use_stemmer=False
    )
    scores = scorer.score(input.reference, input.candidate)

    return {
        "rouge1": {
            "precision": scores['rouge1'].precision,
            "recall": scores['rouge1'].recall,
            "fmeasure": scores['rouge1'].fmeasure
        },
        "rougeL": {
            "precision": scores['rougeL'].precision,
            "recall": scores['rougeL'].recall,
            "fmeasure": scores['rougeL'].fmeasure
        }
    }

@app.post("/score-pdf")
async def calculate_rouge_pdf(resume: UploadFile = File(...), job_description: str = Form(...)):
    resume_text = ""

    with pdfplumber.open(resume.file) as pdf:
        for page in pdf.pages:
            resume_text += page.extract_text() or ""

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rougeL'],
        tokenizer=ThaiTokenizer(),
        use_stemmer=False
    )
    scores = scorer.score(job_description, resume_text)

    return {
        "rouge1": {
            "precision": scores['rouge1'].precision,
            "recall": scores['rouge1'].recall,
            "fmeasure": scores['rouge1'].fmeasure
        },
        "rougeL": {
            "precision": scores['rougeL'].precision,
            "recall": scores['rougeL'].recall,
            "fmeasure": scores['rougeL'].fmeasure
        },
        "resume_text": resume_text
    }

@app.post("/combined-score-pdf")
async def combined_score_pdf(resume: UploadFile = File(...), job_spec: str = Form(...), job_description: str = Form(...)):
    resume_text = ""

    with pdfplumber.open(resume.file) as pdf:
        for page in pdf.pages:
            resume_text += page.extract_text() or ""

    # 1. Semantic similarity job_spec vs resume
    embeddings_spec = semantic_model.encode([job_spec, resume_text])
    semantic_spec_score = float(cosine_similarity([embeddings_spec[0]], [embeddings_spec[1]])[0][0])

    # 2. Semantic similarity job_description vs resume
    embeddings_desc = semantic_model.encode([job_description, resume_text])
    semantic_desc_score = float(cosine_similarity([embeddings_desc[0]], [embeddings_desc[1]])[0][0])

    # 3. ROUGE score
    scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=ThaiTokenizer(), use_stemmer=False)
    rouge_scores = scorer.score(job_spec + " " + job_description, resume_text)
    rouge_score = rouge_scores['rougeL'].fmeasure

    # Combined score: 50% semantic_spec + 30% semantic_desc + 20% ROUGE
    combined_score = (semantic_spec_score * 0.5) + (semantic_desc_score * 0.3) + (rouge_score * 0.2)

    return {
        "combined_score": combined_score,
        "semantic_spec_score": semantic_spec_score,
        "semantic_desc_score": semantic_desc_score,
        "rouge_score": rouge_score,
        "resume_text": resume_text
    }
