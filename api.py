import json
from fastapi import FastAPI,Body
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

class SpeakWordRequest(BaseModel):
    voice_id: str
    word: str
@app.post('/speak_word')
def speak_word(input_params:SpeakWordRequest = Body(...)):
    
    print(input_params)



if __name__ == '__main__':
    uvicorn.run(app='api:app', reload=True, host='0.0.0.0', port=8000)