"""Reference HTTP wrapper for the published Julia Python runtime.

Install Julia from a pinned HF snapshot in its own environment, plus fastapi
and uvicorn. Set JULIA_MODEL_DIR to the downloaded snapshot with real weights.
Start with one process:
    uvicorn julia_service:app --host 127.0.0.1 --port 8008 --workers 1

This example is limited to choice questions. It is not a full TypeSafe server.
It serializes model execution and omits production admission/audit controls.
"""

import asyncio
from contextlib import asynccontextmanager
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=2, max_length=20)

    @field_validator("instructions")
    @classmethod
    def nonblank_instruction(cls, value):
        if not value.strip():
            raise ValueError("Instructions must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def nonblank_options(cls, value):
        if any(not key.strip() or not description.strip()
               for key, description in value.items()):
            raise ValueError("Candidate IDs and descriptions must not be blank")
        return value


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["julia-1"]
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, ChoiceQuestion] = Field(min_length=1, max_length=16)

    @field_validator("questions")
    @classmethod
    def nonblank_question_ids(cls, value):
        if any(not key.strip() for key in value):
            raise ValueError("Question IDs must not be blank")
        return value


@asynccontextmanager
async def lifespan(app):
    # Import only inside the inference environment. Keep the model resident.
    from julia import load_model

    app.state.engine = await asyncio.to_thread(
        load_model, os.environ["JULIA_MODEL_DIR"], device="cpu",
        strict_encoding=True, max_length=8192, head_length=512,
    )
    app.state.inference_lock = asyncio.Lock()
    yield
    del app.state.engine


app = FastAPI(lifespan=lifespan)


@app.post("/v1/systemone")
async def systemone(request: DecisionRequest):
    # The asyncio lock serializes calls without blocking the event loop.
    async with app.state.inference_lock:
        try:
            result = await asyncio.to_thread(
                app.state.engine.predict,
                state=request.state,
                questions={key: value.model_dump()
                           for key, value in request.questions.items()},
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Native Julia probability metadata is preserved; no confidence is invented.
    return result
