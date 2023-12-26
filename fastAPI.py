from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.models import Response


# Load the saved pipeline
pipeline_filename = 'flight_price_pipeline.pkl'
pipeline = joblib.load(pipeline_filename)

# Create a FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Load HTML templates
templates = Jinja2Templates(directory="templates")



# Create a Pydantic model for input data validation
class FlightInput(BaseModel):
    airline: str
    source_city: str
    departure_time: str
    stops: str
    arrival_time: str
    destination_city: str
    class_: str  # Using 'class_' instead of 'class' to avoid keyword conflict
    duration: float
    days_left: int

@app.post("/predict/")
def predict_flight_prices(input_data: FlightInput):
    input_features = pd.DataFrame([dict(input_data)])
    #input_features = list(list(dict(input_data).values()))
    predicted_prices = pipeline.predict(input_features)
    return {"predicted_prices": predicted_prices[0]}


# Define a route to render the HTML form
@app.get("/")
def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

