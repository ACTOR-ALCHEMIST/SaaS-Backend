import base64
import os
from pathlib import Path
from typing import Any, Optional


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional helper
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """
        Minimal fallback loader so local development keeps working even when
        python-dotenv is missing (e.g. running outside the project venv).
        """
        dotenv_path = kwargs.get("dotenv_path")
        if not dotenv_path and args:
            dotenv_path = args[0]
        dotenv_path = Path(dotenv_path or ".env")
        if not dotenv_path.exists():
            print(
                "Warning: python-dotenv is not installed and .env file was not found."
            )
            return False

        loaded = False
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True

        if not loaded:
            print(
                "Warning: python-dotenv is not installed. No environment variables were "
                "loaded from the fallback parser."
            )
        return loaded

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

try:
    from google import genai
    from google.genai import types as genai_types
except ModuleNotFoundError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

try:
    from supabase import Client, create_client
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Client = Any  # type: ignore[assignment]

    def create_client(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Supabase SDK is not installed. Install `supabase` to enable database access."
        )

    print(
        "Warning: Supabase SDK (`supabase` package) not installed. "
        "Database features will be unavailable."
    )

# 加载 .env 文件 (仅用于本地开发)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- 环境变量配置 ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
GENAI_MODEL_ID = os.environ.get("GENAI_MODEL_ID", "gemini-2.5-flash-image")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000").strip()

SERVICE_NAME = "image_generation"
IMAGE_COST = 1

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
    if genai is None:
        raise ModuleNotFoundError(
            "Google GenAI SDK not installed. Install `google-genai` (>=1.0.0)."
        )
    if not GOOGLE_AI_API_KEY:
        raise ValueError("Google AI API key is missing.")
    genai_client = genai.Client(api_key=GOOGLE_AI_API_KEY)
except Exception as exc:  # pragma: no cover - initialization log
    print(f"Error initializing Google GenAI client: {exc}")

app = FastAPI(title="SaaS Backend API")

# --- CORS 中间件 ---
default_origins = {
    FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001",
    "https://fastanswerai.com",
    "https://www.fastanswerai.com",
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
    """Raise a 503 if the image generation client is unavailable."""
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

    # 1. 检查余额
    current_balance: int = 0
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

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Database error for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database processing error: {exc}",
        ) from exc

    # 2. 调用 Gemini
    try:
        print(f"Generating image for user {user_id} prompt: {prompt}")
        contents: list[Any] = [prompt]
        if reference_bytes is not None:
            if genai_types is None or not hasattr(genai_types, "Part"):
                raise RuntimeError(
                    "google-genai types module is unavailable for reference images."
                )
            contents.append(
                genai_types.Part.from_bytes(reference_bytes, mime_type=mime_type)
            )

        response = image_client.models.generate_content(
            model=GENAI_MODEL_ID,
            contents=contents,
        )

        image_data: Optional[str] = None
        alt_text: Optional[str] = None

        def _scan_parts(parts: list[Any]) -> None:
            nonlocal image_data, alt_text
            for part in parts:
                text_val = getattr(part, "text", None)
                if text_val and not alt_text:
                    alt_text = text_val
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data and image_data is None:
                    if isinstance(data, bytes):
                        image_data = base64.b64encode(data).decode("utf-8")
                    else:
                        image_data = data

        def _scan_content(obj: Any) -> None:
            _scan_parts(list(getattr(obj, "parts", []) or []))

        visited: set[int] = set()

        def _walk(obj: Any) -> None:
            if obj is None:
                return
            try:
                marker = id(obj)
            except Exception:
                marker = None
            if marker is not None:
                if marker in visited:
                    return
                visited.add(marker)
            _scan_parts(list(getattr(obj, "parts", []) or []))
            for attr in ("generated_content", "contents", "candidates"):
                items = getattr(obj, attr, None)
                if isinstance(items, list):
                    for child in items:
                        _walk(child)
                elif items is not None:
                    _walk(items)

        _walk(response)

        if image_data is None:
            raise RuntimeError("No image data found in Gemini response.")

        response_id = getattr(response, "response_id", None)
    except Exception as exc:
        print(f"AI generation failed for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI generation failed: {exc}",
        ) from exc

    # 3. 扣费并记录
    try:
        new_balance = max(0, (current_balance or 0) - IMAGE_COST)
        update_res = (
            client.table("wallets")
            .update({"balance": new_balance})
            .eq("user_id", user_id)
            .execute()
        )
        update_error = _response_error_message(update_res)
        if update_error:
            raise RuntimeError(update_error)

        ledger_payload = {
            "user_id": user_id,
            "amount": IMAGE_COST,
            "type": "debit",
            "source": SERVICE_NAME,
            "metadata": {
                "prompt_length": len(prompt),
                "mode": generation_mode,
                "reference_provided": reference_bytes is not None,
            },
        }
        ledger_res = client.table("ledger_entries").insert(ledger_payload).execute()
        ledger_error = _response_error_message(ledger_res)
        if ledger_error:
            raise RuntimeError(ledger_error)

        usage_payload = {
            "user_id": user_id,
            "service": SERVICE_NAME,
            "cost": IMAGE_COST,
            "prompt": prompt,
            "response_id": response_id,
            "metadata": {
                "model": GENAI_MODEL_ID,
                "prompt_length": len(prompt),
                "mode": generation_mode,
                "reference_provided": reference_bytes is not None,
            },
        }
        usage_res = client.table("usage_events").insert(usage_payload).execute()
        usage_error = _response_error_message(usage_res)
        if usage_error:
            print(f"Usage event logging warning for user {user_id}: {usage_error}")

    except Exception as exc:
        print(f"Post-generation bookkeeping failed for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Post-processing failed: {exc}",
        ) from exc

    return {"image_base64": image_data, "alt_text": alt_text or prompt}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
