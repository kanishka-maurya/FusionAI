from peft import PeftModel
from backend.routes.Research_Routes.models.llm_loader import load_quantized_model

def load_finetuned_model():
    tokenizer, base_model = load_quantized_model()

    model = PeftModel.from_pretrained(base_model, "lora-output")

    return tokenizer, model