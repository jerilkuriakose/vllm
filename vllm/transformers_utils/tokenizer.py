import os
import json
import tempfile
import shutil
from typing import Optional, Union

import sentencepiece as spm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.config import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizers import BaichuanTokenizer
from vllm.utils import make_async

logger = init_logger(__name__)


def hf_tokenizer_to_sentencepiece(tokenizer_name):
    new_tokenizer_name = tokenizer_name + "/tokenizer.model"
    tokenizer_config = tokenizer_name + "/tokenizer_config.json"
    tokenizer = spm.SentencePieceProcessor(model_file=new_tokenizer_name)
    with open(tokenizer_config) as f:
        d = json.load(f)
    d['tokenizer_class'] = d['tokenizer_class'].replace('Allam', '')
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename in os.listdir(tokenizer_name):
            if not filename.startswith("model"):
                source_file = os.path.join(tokenizer_name, filename)
                destination_file = os.path.join(tmpdirname, filename)
                shutil.copy(source_file, destination_file)

        temp_fpath = tmpdirname + "/tokenizer_config.json"                
        with open(temp_fpath, 'w') as f:
            json.dump(d, f)

        tokenizer_hf = AutoTokenizer.from_pretrained(tmpdirname)
        
    tokenizer.__dict__.update({
        "all_special_ids": tokenizer_hf.all_special_ids,
        "all_special_tokens_extended": tokenizer_hf.all_special_tokens_extended,
        "all_special_tokens": tokenizer_hf.all_special_tokens,
        "eos_token_id": tokenizer_hf.eos_token_id,
        "is_fast": tokenizer_hf.is_fast,
        "convert_ids_to_tokens": tokenizer_hf.convert_ids_to_tokens,
        "get_added_vocab": tokenizer_hf.get_added_vocab,
        "convert_tokens_to_string": tokenizer_hf.convert_tokens_to_string
    })
    return tokenizer


def get_cached_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer with cached properties.

    This will patch the tokenizer object in place.

    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown. This
    function caches these properties for faster access."""

    tokenizer_all_special_ids = set(tokenizer.all_special_ids)
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)
    tokenizer_all_special_tokens = set(tokenizer.all_special_tokens)
    tokenizer_len = len(tokenizer)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore

        @property
        def all_special_ids(self):
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self):
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self):
            return tokenizer_all_special_tokens_extended

        def __len__(self):
            return tokenizer_len

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    tokenizer.__class__ = CachedTokenizer
    return tokenizer


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface/modelscope."""
    if VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not os.path.exists(tokenizer_name):
            tokenizer_path = snapshot_download(
                model_id=tokenizer_name,
                cache_dir=download_dir,
                revision=tokenizer_revision,
                # Ignore weights - we only need the tokenizer.
                ignore_file_pattern=["*.pt", "*.safetensors", "*.bin"])
            tokenizer_name = tokenizer_path

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs)
    except ValueError as e:
        print(str(e))
        if "LlamaTokenizerAllam" in str(e):
            tokenizer = hf_tokenizer_to_sentencepiece(tokenizer_name)
            
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        elif (not trust_remote_code and
            ("does not exist or is not currently imported." in str(e)
             or "requires you to execute the tokenizer file" in str(e))):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    except AttributeError as e:
        if "BaichuanTokenizer" in str(e):
            # This is for the error "'BaichuanTokenizer' object has no
            # attribute 'sp_model'".
            tokenizer = BaichuanTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return get_cached_tokenizer(tokenizer)


def get_lora_tokenizer(lora_request: LoRARequest, *args,
                       **kwargs) -> Optional[PreTrainedTokenizer]:
    if lora_request is None:
        return None
    try:
        tokenizer = get_tokenizer(lora_request.lora_local_path, *args,
                                  **kwargs)
    except OSError as e:
        # No tokenizer was found in the LoRA folder,
        # use base model tokenizer
        logger.warning(
            f"No tokenizer found in {lora_request.lora_local_path}, "
            "using base model tokenizer instead. "
            f"(Exception: {str(e)})")
        tokenizer = None
    return tokenizer


get_lora_tokenizer_async = make_async(get_lora_tokenizer)
