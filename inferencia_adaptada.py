
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#Variables de entorno
output_dir = "results-with-eval-20epochs"


# Cargando modelo base y adaptador desde la carpeta de salida
base_model_name = "google/gemma-2b-it"
adapter_path = f"./{output_dir}/best_model"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

while True:
    try:
        print("\nIntroduce tu pregunta (o presiona Ctrl+C + ENTER para salir):")
        pregunta = input()
        
        # --- Preparando el Prompt ---
        chat = [
            { "role": "user", "content": f"""Eres un asistente de atención a clientes y estudiantes de la plataforma de educación online en tecnología, inglés y liderazgo llamada Platzi.
        Necesitas responder a la siguiente pregunta: {pregunta}""" },
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # --- Parametros para Generar texto con control adicional ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            # Penaliza la repetición para evitar bucles.
            repetition_penalty=1.2,
        )

        # --- Decodificar y LIMPIAR la respuesta ---
        respuesta_generada = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        # Se limpia el texto basura del final.
        # Busca la posición del último punto, signo de exclamación o interrogación.
        last_punctuation = max(respuesta_generada.rfind('.'), respuesta_generada.rfind('!'), respuesta_generada.rfind('?'))
        
        # Si se encontró un signo de puntuación, cortar la respuesta hasta ese punto.
        if last_punctuation != -1:
            respuesta_limpia = respuesta_generada[:last_punctuation + 1]
        else:
            # Si no hay puntuación, usamos la respuesta original
            respuesta_limpia = respuesta_generada

        print("\n--- Respuesta del Modelo ---")
        print(respuesta_limpia)

    except KeyboardInterrupt:
        print("\n\n¡Hasta luego!")
        break