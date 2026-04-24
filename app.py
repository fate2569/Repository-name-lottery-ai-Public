from fastapi import FastAPI
import numpy as np
import random

app = FastAPI()

def fake_data():
    return np.random.randint(0,10,(200,3))

def freq_model(data):
    count = np.zeros(10)
    for row in data:
        for n in row:
            count[n]+=1
    return np.argsort(count)[-3:][::-1]

@app.get("/predict")
def predict(model:str="freq"):
    data = fake_data()
    if model=="freq":
        res = freq_model(data)
    else:
        res = random.sample(range(10),3)
    return {"result":res}
