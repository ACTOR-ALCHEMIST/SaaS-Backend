from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SaaS Backend API")


class GreetingResponse(BaseModel):
    message: str


class WeatherResponse(BaseModel):
    summary: str
    temperature_c: float


@app.get("/greet", response_model=GreetingResponse)
async def greet(name: str | None = None) -> GreetingResponse:
    """Return a friendly greeting. Defaults to a generic salutation."""
    if name is None:
        greeting = "Hello there!"
    else:
        sanitized = name.strip()
        if not sanitized:
            raise HTTPException(status_code=400, detail="Name cannot be empty.")
        greeting = f"Hello {sanitized}, I think you are great!"
    return GreetingResponse(message=greeting)


@app.get("/weather/today", response_model=WeatherResponse)
async def fetch_weather_today() -> WeatherResponse:
    """Provide a placeholder weather report for today."""
    # TODO: Replace with a real weather service integration.
    return WeatherResponse(summary="Sunny with light breeze", temperature_c=23.5)
