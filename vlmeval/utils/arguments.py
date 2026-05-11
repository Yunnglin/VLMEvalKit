from dataclasses import dataclass, field
from typing import Optional, List, Union
from os import environ
from vlmeval.smp import *

logger = get_logger("arguments")


def _parse_reuse_aux(reuse_aux) -> str:
    """Normalize reuse_aux to one of 'all' / 'infer' / 'none'.

    Accepts bool / int for backward compatibility.
    """
    if isinstance(reuse_aux, bool):
        return 'all' if reuse_aux else 'none'
    if isinstance(reuse_aux, int):
        return 'all' if reuse_aux else 'none'
    if isinstance(reuse_aux, str):
        value = reuse_aux.strip().lower()
        if value in ['all', 'infer', 'none']:
            return value
        if value in ['1', 'true', 'yes']:
            return 'all'
        if value in ['0', 'false', 'no']:
            return 'none'
    raise ValueError(f"reuse_aux must be one of: all, infer, none; got {reuse_aux!r}")


@dataclass
class Arguments:
    # ---- Essential Args ----
    data: List[str] = field(default_factory=list)
    model: Union[List[dict], List[str]] = field(default_factory=list)
    config: Optional[str] = None

    # ---- Work Dir & Mode ----
    work_dir: str = "outputs"
    mode: str = "all"

    # ---- API Kwargs ----
    api_nproc: int = 4
    retry: int = 6
    verbose: bool = False
    keep_failed: bool = False
    ignore: bool = False  # Deprecated: default behavior now; use keep_failed=True to disable
    reuse: bool = False
    reuse_aux: Union[str, bool] = 'all'  # 'all' | 'infer' | 'none'; bool accepted for backward compat
    use_vllm: bool = False
    use_verifier: bool = False

    # ---- Judge Args ----
    judge: Optional[str] = None
    judge_args: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_key: Optional[str] = None
    judge_api_nproc: Optional[int] = None
    judge_retry: Optional[int] = None
    judge_timeout: int = 600

    # ---- Inference Model Args (effective when base_url is set) ----
    base_url: Optional[str] = None
    key: str = 'sk-admin'
    thinker: bool = False  # Deprecated: use max_tokens / timeout directly
    max_tokens: int = 2 ** 15
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    timeout: int = 1800
    custom_prompt: Optional[str] = None
    extra_body: Optional[str] = None
    video_llm: bool = False
    local_media: bool = False

    # ---- API Pipeline Args (only for api_mode) ----
    api_mode: bool = False
    monitor_interval: int = 30
    debug: bool = False

    # ---- Legacy Args (kept for backward compatibility) ----
    fps: int = -1
    nframe: int = 8
    pack: bool = False
    use_subtitle: bool = False
    nproc: Optional[int] = None  # Deprecated: use api_nproc instead; if set, overrides api_nproc
    limit: Optional[int] = None

    # ---- Legacy OpenAI API Args ----
    OPENAI_API_KEY: str = "EMPTY"
    OPENAI_API_BASE: Optional[str] = None
    LOCAL_LLM: Optional[str] = None

    def __post_init__(self):
        # Normalize reuse_aux: accept bool/int for backward compatibility
        try:
            self.reuse_aux = _parse_reuse_aux(self.reuse_aux)
        except ValueError as e:
            logger.error(f"Invalid reuse_aux value: {e}")
            raise

        # Sync deprecated nproc -> api_nproc
        if self.nproc is not None:
            logger.warning(
                '[Deprecated] The `nproc` field is deprecated, use `api_nproc` instead. '
                f'Setting api_nproc={self.nproc}.'
            )
            self.api_nproc = self.nproc

        # Warn about deprecated fields
        if self.ignore:
            logger.warning(
                '[Deprecated] The `ignore` field is deprecated since ignoring failed indices '
                'is the default behavior. Use `keep_failed=True` to disable it.'
            )
        if self.thinker:
            logger.warning(
                '[Deprecated] The `thinker` field is deprecated. '
                'Use `max_tokens` and `timeout` directly instead.'
            )

        # Set environment variables for legacy OpenAI API
        try:
            if self.OPENAI_API_BASE and self.LOCAL_LLM:
                environ.update(
                    {
                        "OPENAI_API_KEY": self.OPENAI_API_KEY,
                        "OPENAI_API_BASE": self.OPENAI_API_BASE,
                        "LOCAL_LLM": self.LOCAL_LLM,
                    }
                )
        except Exception as e:
            logger.error(f"Error occurred when setting environment variables: {e}")
            raise e
