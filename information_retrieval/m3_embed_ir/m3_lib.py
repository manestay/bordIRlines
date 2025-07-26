from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path


@dataclass
class ModelArgs:
    encoder: str = field(default="BAAI/bge-m3", metadata={"help": "Name or path of encoder"})
    pooling_method: str = field(
        default="cls", metadata={"help": "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "Normalize embeddings or not"}
    )
    fp16: bool = field(default=True, metadata={"help": "Use fp16 in inference?"})


@dataclass
class EvalArgs:
    languages: str = field(
        default=("all",), metadata={"help": "Languages to evaluate.", "nargs": "*"}
    )
    index_save_dir: Path = field(
        default=Path("./corpus-index"),
        metadata={
            "help": "Dir to index and docid. Corpus index path is `index_save_dir/{encoder_name}/index`. Corpus ids path is `index_save_dir/{encoder_name}/docid` ."
        },
    )
    result_save_dir: Path = field(
        default="./search_results",
        metadata={
            "help": "Dir to saving search results. Search results will be saved to `result_save_dir/{encoder_name}/{lang}.txt`"
        },
    )
    threads: int = field(default=16, metadata={"help": "Maximum threads to use during search"})
    batch_size: int = field(default=256, metadata={"help": "Search batch size."})
    hits: int = field(default=100, metadata={"help": "Number of hits"})
    overwrite: bool = field(default=False, metadata={"help": "Whether to overwrite embedding"})
    retrieval_over: Literal["qlang", "qlang_en", "en", "rel_langs"] = "qlang"
