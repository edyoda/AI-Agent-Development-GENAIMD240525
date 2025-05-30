from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Any
import uvicorn

app = FastAPI()

class CallbackPayload(BaseModel):
    status: str
    course: str
    topic: str
    sub_topics: List[str]
    meta_id: str
    result: Any  # Accepts large text or structured response

@app.post("/callbackassignments/")
async def callback_assignments(payload: CallbackPayload):
    print("\n=== Webhook Callback Received ===")
    print(f"Status    : {payload.status}")
    print(f"Course    : {payload.course}")
    print(f"Topic     : {payload.topic}")
    print(f"Sub-topics: {payload.sub_topics}")
    print(f"Meta ID   : {payload.meta_id}")
    print("\n--- Assignments Result ---")
    print(payload.result)
    print("--------------------------\n")
    return {"message": "Callback received successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

