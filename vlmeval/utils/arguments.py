from dataclasses import dataclass, field
from typing import Optional, List, Union
from os import environ


@dataclass
class Arguments:
    data: List[str] = field(default_factory=list)
    model: Union[List[dict], List[str]] = field(default_factory=dict)
    nframe: int = 8
    pack: bool = False
    use_subtitle: bool = False
    work_dir: str = '.'
    mode: str = 'all'
    nproc: int = 1
    retry: Optional[int] = None
    judge: Optional[str] = None
    verbose: bool = False
    ignore: bool = False
    rerun: bool = False
    limit: Optional[int] = None

    # For OpenAI API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    LOCAL_LLM: Optional[str] = None

    def __post_init__(self):
        environ.update({
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'OPENAI_API_BASE': self.OPENAI_API_BASE,
            'LOCAL_LLM': self.LOCAL_LLM
        })
