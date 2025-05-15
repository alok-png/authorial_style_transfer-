from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

class Prompt(BaseModel):
    inputs: str

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model pipeline once
generator = pipeline("text-generation", model="alok1108/authorial_style_transfer_jane")

@app.post("/generate")
def generate(prompt: Prompt):
    result = generator(prompt.inputs, max_length=100, do_sample=True, top_p=0.9, temperature=0.8)
    return {"output": result[0]["generated_text"]}
