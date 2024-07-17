from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class Arguments:
    data: List[str] = field(default_factory=list)
    model: List[str] = field(default_factory=list)
    nframe: int = 8
    pack: bool = False
    work_dir: str = '.'
    mode: str = 'all'
    nproc: int = 1
    retry: Optional[int] = None
    judge: Optional[str] = None
    verbose: bool = False
    ignore: bool = False
    rerun: bool = False
    limit: Optional[int] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    LOCAL_LLM: Optional[str] = None