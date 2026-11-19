import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = Path(os.environ.get("GAN_WEIGHTS_PATH", BASE_DIR / "gan_weights_99.npz"))
GENERATOR_SHAPES = ((100, 256), (256, 1024), (1024, 512), (512, 784))
generator_weights: list[tuple[np.ndarray, np.ndarray]] | None = None


def load_generator_weights(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    weights = []
    with np.load(path, allow_pickle=False) as archive:
        for index, (input_size, output_size) in enumerate(GENERATOR_SHAPES, start=1):
            weight_key = f"G_w{index}"
            bias_key = f"G_b{index}"
            if weight_key not in archive or bias_key not in archive:
                raise ValueError(f"Checkpoint is missing {weight_key} or {bias_key}")

            matrix = np.asarray(archive[weight_key], dtype=np.float32)
            bias = np.asarray(archive[bias_key], dtype=np.float32)
            if matrix.shape != (input_size, output_size):
                raise ValueError(
                    f"{weight_key} has shape {matrix.shape}; "
                    f"expected {(input_size, output_size)}"
                )
            if bias.shape not in ((output_size,), (1, output_size)):
                raise ValueError(
                    f"{bias_key} has shape {bias.shape}; "
                    f"expected {(1, output_size)}"
                )
            if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
                raise ValueError(f"{weight_key} and {bias_key} must contain finite values")
            weights.append((matrix, bias.reshape(1, output_size)))
    return weights


@asynccontextmanager
async def lifespan(_: FastAPI):
    global generator_weights
    try:
        generator_weights = load_generator_weights(WEIGHTS_PATH)
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"Unable to load GAN generator weights from {WEIGHTS_PATH}: {error}") from error
    yield
    generator_weights = None


app = FastAPI(title="MNIST GAN Pixel Lab API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    seed: int = Field(default=1592598566, ge=0, le=4294967295)
    latent_scale: float = Field(default=1.0, ge=0.1, le=2.5)
    gamma: float = Field(default=1.0, ge=0.4, le=2.5)
    activation: Literal["tanh", "sigmoid"] = "tanh"


class GenerateResponse(BaseModel):
    seed: int
    pixels: list[float]
    minimum: float
    maximum: float
    mean: float


@app.get("/")
def root():
    return {"status": "ready", "service": "MNIST GAN Pixel Lab API"}


@app.get("/health")
def health():
    return {"status": "ready", "weights_loaded": generator_weights is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate_sample(request: GenerateRequest):
    if generator_weights is None:
        raise HTTPException(status_code=503, detail="Generator weights are not loaded")

    latent = np.random.default_rng(request.seed).standard_normal((1, 100)).astype(np.float32)
    latent *= request.latent_scale
    output = latent
    for index, (matrix, bias) in enumerate(generator_weights):
        output = output @ matrix + bias
        if index == 1:
            mean = output.mean(axis=1, keepdims=True)
            variance = output.var(axis=1, keepdims=True)
            output = (output - mean) / np.sqrt(variance + 1e-8)
        if index < len(generator_weights) - 1:
            output = np.where(output > 0, output, 0.2 * output)

    if request.activation == "tanh":
        pixels = (np.tanh(output) + 1) / 2
    else:
        clipped = np.clip(output, -60, 60)
        pixels = 1 / (1 + np.exp(-clipped))
    pixels = np.power(np.clip(pixels, 0, 1), request.gamma).reshape(784)

    return GenerateResponse(
        seed=request.seed,
        pixels=pixels.tolist(),
        minimum=float(pixels.min()),
        maximum=float(pixels.max()),
        mean=float(pixels.mean()),
    )
