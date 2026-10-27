import os

from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


def _get_config_id() -> str:
    config_id = settings.PORTKEY_CONFIG_ID

    if not config_id:
        raise RuntimeError("PORTKEY_CONFIG_ID is not configured")

    config_id = config_id.strip()

    if not config_id.startswith("pc-"):
        raise RuntimeError(
            "PORTKEY_CONFIG_ID must be a saved Portkey config ID "
            "starting with 'pc-'"
        )

    return config_id


def _validate_gateway_settings() -> None:
    if not settings.PORTKEY_API_KEY:
        raise RuntimeError("PORTKEY_API_KEY is not configured")

    _get_config_id()


# Validate configuration before creating either client.
_validate_gateway_settings()

# Native Portkey client: used by responder.py.
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=_get_config_id(),
)


# LangChain client: used by planner.py and other agent nodes.
def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    headers = createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config=_get_config_id(),
        metadata={"feature": feature},
    )

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model="openai/gpt-oss-120b",
        temperature=0,
        default_headers=headers,
    )


def extract_cache_status(response) -> str:
    # Native Portkey SDK responses may expose headers differently
    # depending on the SDK version.
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)

        if raw is not None:
            headers = getattr(raw, "headers", {})
            status = headers.get("x-portkey-cache-status", "")

            if status:
                return status.upper()

    return "UNKNOWN"
