
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.auth.routes import router as auth_router
from backend.datasets.routes import router as dataset_router
from backend.api.forecasts import router as forecast_router
from backend.api.bottleneck import router as bottleneck_router
from backend.api.dashboard import router as dashboard_router
from backend.api.weather import router as weather_router


app = FastAPI(title="CareFlow API")


# CORS configuration: localhost by default, configurable for production.
default_origins = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        default_origins,
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(
    request: Request,
    exc: SQLAlchemyError,
):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "A database error occurred while processing the request."
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred."
        },
    )


app.include_router(auth_router)
app.include_router(dataset_router)
app.include_router(forecast_router)
app.include_router(bottleneck_router)
app.include_router(dashboard_router)
app.include_router(weather_router)


@app.get("/")
def root():
    return {"message": "CareFlow API is running"}
