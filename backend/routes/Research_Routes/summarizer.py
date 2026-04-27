import json
PROMPT_TEMPLATE = """

Summarize the following text into JSON:

- summary: short explanation containing relevant details of whole text in few sentences
- entities: important names/concepts
- risks: potential issues

TEXT:
{chunk}

OUTPUT:
"""

def summarize_chunk(chunk, tokenizer, model):
    prompt = PROMPT_TEMPLATE.format(chunk=chunk)
    print(model.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_start = response.find("{")
        parsed = json.loads(response[json_start:])
    except:
        parsed = {
            "summary": response,
            "entities": [],
            "risks": []
        }

    return parsed