from fastapi import FastAPI, File, UploadFile, Request
import pandas as pd
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object
from src.utils.model_utils.model_utils import NetworkModel
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home():
    return {"message": "Docs"}

@app.get("/train")
async def train():
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()
    return {"message": "Training Completed"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    model = load_object('final_objects/model.pkl')
    preprocessor = load_object('final_objects/preprocessing.pkl')
    df = pd.read_csv(file.file)
    network_model = NetworkModel(preprocessor=preprocessor, model=model)
    predictions = network_model.predict(df)
    df['predictions'] = predictions
    table = df.to_html()
    return templates.TemplateResponse("table.html", {"request": request, "table": table})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)