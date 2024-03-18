from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from logging import handlers
from intelliscript import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s|%(message)s')
handler = handlers.TimedRotatingFileHandler(
    "logs/intelliscript.log", when="H", interval=24)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    id: int
    query: str


@app.post("/query")
async def user_query(user_query: Query):
    logger.info(f'User query: {user_query.id}-{user_query.query}')
    response = response_generation(user_query.query)
    logger.info(f'LLM response: {user_query.id}-{response}')
    return {'id': user_query.id, 'content': response}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
