# Model architectures
#
# NOTE: `.model` and `.model_2` are legacy modules with a broken import
# (`from ..core.transformer import TransformerBlock` — symbol no longer exists).
# They are imported defensively so a broken legacy module does not take down the
# whole `src.models` package for the working architectures below.
try:  # pragma: no cover - legacy, may be broken
    from .model import CoreModel
except Exception:  # noqa: BLE001
    CoreModel = None
try:  # pragma: no cover - legacy, may be broken
    from .model_2 import CoreModel2
except Exception:  # noqa: BLE001
    CoreModel2 = None

from .core_outline_model import CoreOutlineForCausalLM, CoreOutlineConfig, create_coreoutline_qwen_model
from .core_model import (
    GatedCoreModel,
    CoreModelForCausalLM,
    CoreModelConfig,
    create_core_model,
)
from .bart_mnli import BartForSequenceClassification, BartMnliConfig
from .tool_selection import (
    ZeroShotToolSelector,
    ToolScore,
    load_tool_selector,
    run_tool_arm,
    detect_tool_gate,
    parse_tools,
)
