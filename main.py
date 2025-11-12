import base64
import os
from typing import Any, Optional

try:
    from google import genai as google_genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    USING_GENAI_CLIENT = True
except ImportError:  # pragma: no cover
    import google.generativeai as google_genai  # type: ignore
    from google.generativeai import types as genai_types  # type: ignore
    USING_GENAI_CLIENT = False
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client, create_client

# 加载 .env 文件 (仅用于本地开发)
load_dotenv()

# --- 环境变量配置 ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000").strip()

SERVICE_NAME = "image_generation"
IMAGE_COST = 1
MODEL_NAME = "gemini-2.5-flash-image"

# --- 服务客户端初始化 ---
supabase: Optional[Client] = None
try:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase URL or service key is missing.")
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as exc:  # pragma: no cover - initialization log
    print(f"Error initializing Supabase: {exc}")

genai_client: Optional[Any] = None
try:
    if not GOOGLE_AI_API_KEY:
        raise ValueError("Google AI API key is missing.")
    if USING_GENAI_CLIENT:
        genai_client = google_genai.Client(api_key=GOOGLE_AI_API_KEY)
    else:
        google_genai.configure(api_key=GOOGLE_AI_API_KEY)
        genai_client = google_genai.GenerativeModel(MODEL_NAME)
except Exception as exc:  # pragma: no cover - initialization log
    print(f"Error initializing Google AI: {exc}")

app = FastAPI(title="SaaS Backend API")

# --- CORS 中间件 ---
default_origins = {
    FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001",
    "https://fastanswerai.com",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(default_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GreetingResponse(BaseModel):
    message: str


class WeatherResponse(BaseModel):
    summary: str
    temperature_c: float


async def _ensure_supabase() -> Client:
    """Raise a 503 if Supabase is unavailable."""
    if supabase is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase client is not configured.",
        )
    return supabase


async def _ensure_genai_client() -> Any:
    """Raise a 503 if the Gemini client is unavailable."""
    if genai_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image generation client is not configured.",
        )
    return genai_client


def _response_error_message(response: Any) -> Optional[str]:
    """Extract an error message from a Supabase response object, if any."""
    error = getattr(response, "error", None)
    if not error:
        return None
    return getattr(error, "message", str(error))


def _decode_base64_payload(payload: str) -> bytes:
    """Decode optional data URI / base64 payload into raw bytes."""
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _make_text_part(text: str) -> Any:
    """Create a text Part compatible with both new and legacy Gemini SDKs."""
    part_type = getattr(genai_types, "Part", None)
    part_factory = getattr(part_type, "from_text", None) if part_type else None
    if callable(part_factory):
        return part_factory(text)
    return {"text": text}


def _make_image_part(image_bytes: bytes, mime_type: str) -> Any:
    """Create an image Part compatible with both new and legacy Gemini SDKs."""
    part_type = getattr(genai_types, "Part", None)
    part_factory = getattr(part_type, "from_bytes", None) if part_type else None
    if callable(part_factory):
        return part_factory(image_bytes, mime_type=mime_type)
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_bytes,
        }
    }


def _make_content(parts: list[Any]) -> Any:
    """Wrap parts into a Content structure compatible with both SDKs."""
    content_cls = getattr(genai_types, "Content", None)
    if content_cls is not None:
        try:
            return content_cls(parts=parts)
        except TypeError:
            return content_cls(role="user", parts=parts)
    return {"role": "user", "parts": parts}


# --- 依赖项：JWT 验证 ---
async def get_user_id(request: Request) -> str:
    """
    从请求头中获取 Supabase JWT, 验证它, 并返回 user_id。
    这是我们的API安全守卫。
    """
    client = await _ensure_supabase()

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    try:
        token_parts = auth_header.split(" ")
        if len(token_parts) != 2:
            raise ValueError("Malformed Authorization header.")
        token = token_parts[1]
        user_response = client.auth.get_user(token)
        user = getattr(user_response, "user", None)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        return user.id

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Auth error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        ) from exc


@app.get("/")
async def health_check() -> dict[str, str]:
    """
    一个简单的健康检查端点，确保服务正在运行。
    """
    return {
        "status": "ok",
        "service": app.title,
        "message": "AIGC Backend is running!",
    }


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


@app.post("/api/v1/generate")
async def generate_image(
    prompt: str = Form(...),
    reference_image_base64: str | None = Form(None),
    reference_image_mime_type: str | None = Form(None),
    user_id: str = Depends(get_user_id),
) -> dict[str, str]:
    """
    核心API：生成图片
    1. 验证用户 (已通过 Depends(get_user_id) 完成)
    2. 检查并扣除积分
    3. 调用 AI API
    4. 返回结果
    """
    client = await _ensure_supabase()
    image_client = await _ensure_genai_client()

    reference_bytes: Optional[bytes] = None
    if reference_image_base64:
        try:
            reference_bytes = _decode_base64_payload(reference_image_base64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reference image payload.",
            ) from exc

    mime_type = (reference_image_mime_type or "image/png").strip() or "image/png"
    generation_mode = "edit" if reference_bytes else "generate"

    # 1. 检查并扣费
    try:
        wallet_res = (
            client.table("wallets")
            .select("balance")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        status_code = getattr(wallet_res, "status_code", None)
        if status_code == 406:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Wallet not found for user.",
            )

        error_message = _response_error_message(wallet_res)
        if error_message:
            raise RuntimeError(error_message)

        wallet = getattr(wallet_res, "data", None) or {}
        current_balance = wallet.get("balance")

        if current_balance is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to determine wallet balance.",
            )

        if current_balance < IMAGE_COST:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Insufficient credits",
            )

        ledger_payload = {
            "user_id": user_id,
            "amount": IMAGE_COST,
            "type": "debit",
            "source": SERVICE_NAME,
            "metadata": {
                "prompt_length": len(prompt),
                "mode": generation_mode,
            },
        }
        ledger_res = client.table("ledger_entries").insert(ledger_payload).execute()
        ledger_error = _response_error_message(ledger_res)
        if ledger_error:
            raise RuntimeError(ledger_error)

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Database error for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database processing error: {exc}",
        ) from exc

    # 2. 调用 AI
    try:
        print(f"Generating image for user {user_id} prompt: {prompt}")
        if USING_GENAI_CLIENT:
            contents: list[Any] = [prompt]
            if reference_bytes is not None:
                contents.append(_make_image_part(reference_bytes, mime_type))

            response = image_client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
        else:
            if reference_bytes is None:
                response = image_client.generate_content(prompt)
            else:
                parts = [
                    _make_text_part(prompt),
                    _make_image_part(reference_bytes, mime_type),
                ]
                response = image_client.generate_content([_make_content(parts)])

        parts = getattr(response, "parts", None) or []
        if not parts and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
        image_part = next(
            (part for part in parts if getattr(part, "inline_data", None)), None
        )
        text_part = next((part for part in parts if getattr(part, "text", None)), None)

        if not image_part or not getattr(image_part.inline_data, "data", None):
            raise RuntimeError("No image data found in Gemini response.")

        image_data = image_part.inline_data.data
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode("utf-8")
        text_data = getattr(text_part, "text", None) if text_part else None

        usage_payload = {
            "user_id": user_id,
            "service": SERVICE_NAME,
            "cost": IMAGE_COST,
            "prompt": prompt,
            "response_id": getattr(response, "response_id", None),
            "metadata": {
                "model": MODEL_NAME,
                "prompt_length": len(prompt),
                "mode": generation_mode,
                "reference_provided": reference_bytes is not None,
            },
        }
        usage_res = client.table("usage_events").insert(usage_payload).execute()
        usage_error = _response_error_message(usage_res)
        if usage_error:
            print(f"Usage event logging warning for user {user_id}: {usage_error}")

        return {"image_base64": image_data, "alt_text": text_data or prompt}

    except Exception as exc:
        print(f"AI generation failed for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI generation failed: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
