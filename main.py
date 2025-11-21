from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from rouge_score import rouge_scorer
from pydantic import BaseModel
from pythainlp import word_tokenize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
