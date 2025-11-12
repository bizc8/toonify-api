from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from toon import encode as json_to_toon_encode
from toon import decode as toon_to_json_decode
import json
import tiktoken

app = FastAPI(title="JSON ↔ TOON Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JSONPayload(BaseModel):
    data: dict

class TOONPayload(BaseModel):
    data: str

def token_count(text: str, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except:
        # fallback
        return len(text.split())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/convert/json-to-toon")
def json_to_toon(payload: JSONPayload):
    try:
        toon_output = json_to_toon_encode(payload.data)
        return {
            "toon": toon_output,
            "tokens_original": token_count(json.dumps(payload.data)),
            "tokens_toon": token_count(toon_output),
            "reduction_percent": round(
                (1 - token_count(toon_output) / token_count(json.dumps(payload.data))) * 100,
                2
            )
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/convert/toon-to-json")
def toon_to_json(payload: TOONPayload):
    try:
        decoded_json = toon_to_json_decode(payload.data)
        return {"json": decoded_json}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize")
def optimize(payload: JSONPayload):
    try:
        original_json = json.dumps(payload.data, indent=2)
        toon_output = json_to_toon_encode(payload.data)

        original_tokens = token_count(original_json)
        toon_tokens = token_count(toon_output)

        return {
            "original_json": original_json,
            "toon": toon_output,
            "stats": {
                "original_tokens": original_tokens,
                "toon_tokens": toon_tokens,
                "reduction_percent": round((1 - toon_tokens / original_tokens) * 100, 2),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
