# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .llm_api import LLMAPI, ToolList, ToolParserName
from .core import LLMResult
from .manual_api import ManualAPI
from .core import ToolCall
from .. import setup_lazy_imports, TYPE_CHECKING

# Register submodules with optional dependencies.
if TYPE_CHECKING:
    from .openai_api import OpenAIAPI
    from .gemini_api import GeminiAPI
    from .deepseek_api import DeepSeekAPI
    from .openrouter_api import OpenRouterAPI
    from .siliconflow_api import SiliconFlowAPI
__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    # Optional clients are imported only when their dependencies are present.
    "OpenAIAPI": (".openai_api", "all"),
    "GeminiAPI": (".gemini_api", "all"),
    "DeepSeekAPI": (".deepseek_api", "all"),
    "OpenRouterAPI": (".openrouter_api", "all"),
    "SiliconFlowAPI": (".siliconflow_api", "all"),
})
