from fastapi import FastAPI
from api.routers import styletransferapi, faststyletransfer


app = FastAPI(
    title="Style Transfer API",
    description="An API to apply style transfer to images",
    version="1.0.0"
)


# Include routers
app.include_router(styletransferapi.router, prefix="/api/v1")
app.include_router(faststyletransfer.router, prefix="/api/v1")

# Run the app with uvicorn: uvicorn api.main_api:app --reload