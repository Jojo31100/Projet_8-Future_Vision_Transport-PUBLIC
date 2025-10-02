from fastapi import FastAPI

app = FastAPI(title="API Test Ultra Simple")

@app.get("/")
async def root():
    return {"message": "Site en ligne"}