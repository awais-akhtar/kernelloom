"""LangChain adapter for a resident KernelLoom model."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
except ImportError as exc:  # pragma: no cover - exercised when the optional extra is absent
    raise ImportError(
        "LangChain support is optional. Install KernelLoom with 'pip install kernelloom[langchain]'."
    ) from exc

from .model import KernelLoomModel


class KernelLoomChatModel(BaseChatModel):
    """Use KernelLoom anywhere LangChain accepts a chat model."""

    model: Any

    def __init__(self, model: KernelLoomModel, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "kernelloom"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model.config.model_id, "backend": self.model.config.resolved_backend}

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self.model.chat(_convert_messages(messages), stop_strings=stop, **kwargs)
        message = AIMessage(content=result.text, response_metadata=result.metadata)
        return ChatResult(generations=[ChatGeneration(message=message)])

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


def _convert_messages(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return [
        {"role": roles.get(message.type, message.type), "content": str(message.content)}
        for message in messages
    ]
