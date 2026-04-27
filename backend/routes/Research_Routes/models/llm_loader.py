import multiprocessing

_model = None

def load_model_once():
    global _model

    if _model is None:
        print("Loading GGUF Qwen model (only once)...")

        from ctransformers import AutoModelForCausalLM

        cpu_threads = multiprocessing.cpu_count()  

        _model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Qwen2.5-7B-Instruct-GGUF",   
            model_file="qwen2.5-7b-instruct.Q4_K_M.gguf",
            model_type="qwen",
            threads=cpu_threads,   
            gpu_layers=0,          
            context_length=2048,   
            batch_size=8,         
        )

        print("Model loaded successfully!")

    return _model