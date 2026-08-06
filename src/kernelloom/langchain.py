"""First-class LangChain adapters for local chat and embeddings."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, AsyncIterator, Iterator, Sequence

try:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.output_parsers.openai_tools import parse_tool_calls
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as exc:  # pragma: no cover - exercised when the optional extra is absent
    raise ImportError(
        "LangChain support is optional. Install KernelLoom with 'pip install kernelloom[langchain]'."
    ) from exc

from .config import ModelConfig
from .model import GenerationResult, KernelLoomModel


class KernelLoomChatModel(BaseChatModel):
    """Use one resident local model in chains, agents, and async applications."""

    model: Any

    def __init__(self, model: KernelLoomModel, **kwargs: Any) -> None:
        kwargs.setdefault("disable_streaming", "tool_calling")
        super().__init__(model=model, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "kernelloom"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model.config.model_id,
            "model_path": self.model.config.model_path,
            "backend": self.model.config.resolved_backend,
            "context_length": self.model.config.context_length,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Bind tools using llama.cpp's local tool-aware chat templates."""

        if self.model.config.resolved_backend != "llama-cpp":
            raise NotImplementedError("Tool calling currently requires a tool-capable GGUF chat template")
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if not formatted:
            raise ValueError("tools cannot be empty")
        choice: str | dict[str, Any] | None = tool_choice
        if tool_choice in {"any", "required"}:
            choice = "required"
        elif tool_choice and tool_choice not in {"auto", "none"}:
            names = {tool["function"]["name"] for tool in formatted}
            if tool_choice not in names:
                raise ValueError(f"Unknown tool_choice: {tool_choice}")
            choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(tools=formatted, tool_choice=choice, **kwargs)

    def close(self) -> None:
        self.model.close()

    def get_num_tokens(self, text: str) -> int:
        return self.model.count_tokens(text)

    def __enter__(self) -> "KernelLoomChatModel":
        self.model.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self.model.chat(_convert_messages(messages), stop_strings=stop, **kwargs)
        return _chat_result(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await self.model.achat(_convert_messages(messages), stop_strings=stop, **kwargs)
        return _chat_result(result)

    def _stream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for text in self.model.stream(_convert_messages(messages), stop_strings=stop, **kwargs):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for text in self.model.astream(_convert_messages(messages), stop_strings=stop, **kwargs):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                await run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


class KernelLoomEmbeddings(Embeddings):
    """LangChain embeddings backed by a local sequence-embedding GGUF model."""

    def __init__(self, model: KernelLoomModel | ModelConfig | str, **options: Any) -> None:
        if isinstance(model, KernelLoomModel):
            if not model.config.embedding:
                raise ValueError("The supplied KernelLoomModel must use embedding=True")
            self.model = model
        elif isinstance(model, ModelConfig):
            self.model = KernelLoomModel(replace(model, embedding=True))
        else:
            self.model = KernelLoomModel(ModelConfig(model, embedding=True, **options))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_many(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.model.aembed_many(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.model.aembed(text)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> "KernelLoomEmbeddings":
        self.model.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _chat_result(result: GenerationResult) -> ChatResult:
    usage = result.metadata.get("usage", {})
    usage_metadata = None
    if usage:
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens)),
        }
    raw_calls = result.metadata.get("tool_calls", [])
    tool_calls = parse_tool_calls(raw_calls, return_id=True) if raw_calls else []
    metadata = {
        **result.metadata,
        "model_id": result.model_id,
        "backend": result.backend,
        "device": result.device,
        "latency_ms": result.latency_ms,
    }
    message = AIMessage(
        content=result.text,
        response_metadata=metadata,
        tool_calls=tool_calls,
        usage_metadata=usage_metadata,
    )
    generation = ChatGeneration(
        message=message,
        generation_info={"finish_reason": result.metadata.get("finish_reason")},
    )
    return ChatResult(generations=[generation], llm_output={"model_id": result.model_id, "usage": usage})


def _convert_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    for message in messages:
        item: dict[str, Any] = {
            "role": roles.get(message.type, message.type),
            "content": _text_content(message.content),
        }
        if getattr(message, "name", None):
            item["name"] = message.name
        if message.type == "tool" and getattr(message, "tool_call_id", None):
            item["tool_call_id"] = message.tool_call_id
        calls = getattr(message, "tool_calls", None)
        if calls:
            item["tool_calls"] = [
                {
                    "id": call.get("id", ""),
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
                for call in calls
            ]
        converted.append(item)
    return converted


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "plain_text"}:
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)
