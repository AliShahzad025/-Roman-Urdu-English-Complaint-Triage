"""Loading the small language model (optionally with the LoRA adapter) and batched generation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import BASE_MODEL, EVAL_BATCH_SIZE, MAX_NEW_TOKENS
from prompts import messages_for


def compute_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # T4 -> float16
    return torch.float32


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(for_training: bool = False):
    """4-bit NF4 on a GPU (QLoRA); full precision on a CPU (for smoke tests only).

    The base and fine-tuned models are both evaluated in 4-bit, so quantisation does not bias the comparison."""
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype())
        kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=compute_dtype(), **kwargs)
    except TypeError:   # older transformers
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=compute_dtype(), **kwargs)
    if not for_training:
        model.eval()
    return model


def load_model(adapter_dir=None):
    model = load_base_model()
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model


@torch.no_grad()
def generate(model, tok, texts: list[str], batch_size: int = EVAL_BATCH_SIZE) -> list[str]:
    tok.padding_side = "left"   # decoder-only models must be left-padded for batched generation
    outputs = []
    for start in range(0, len(texts), batch_size):
        prompts = [tok.apply_chat_template(messages_for(t), tokenize=False, add_generation_prompt=True)
                   for t in texts[start:start + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outputs
