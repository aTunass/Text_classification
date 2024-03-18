from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from module import Translation
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
translater = Translation()
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
tokenizer = AutoTokenizer.from_pretrained("aTunass/distilbert-base-uncased-finetuned-Tweets_IMDB_dataset-finetuned-text_classification")
model = AutoModelForSequenceClassification.from_pretrained("aTunass/distilbert-base-uncased-finetuned-Tweets_IMDB_dataset-finetuned-text_classification",
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
@app.get("/text_classification")
async def text_search(text: str = Query(..., description="Search query")):
    text = translater(text)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    result = model.config.id2label[predicted_class_id]
    return JSONResponse(content={'results': result})
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
