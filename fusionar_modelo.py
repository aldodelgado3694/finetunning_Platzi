# fusionar_modelo.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#Variables de entorno
output_dir = "results-with-eval-20epochs"

# --- Carga el modelo y el adaptador ---
base_model_name = "google/gemma-2b-it"
adapter_path = f"./{output_dir}/best_model"
# La nueva carpeta donde guardarás tu modelo fusionado
merged_model_path = "./gemma-2b-it-platzibot" 

# --- Cargar el modelo base y el adaptador ---

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    return_dict=True,
    low_cpu_mem_usage=True
)

# --- Aplicando adaptador LoRA ---
model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)

# --- Fusionar y descargar ---
# El método `merge_and_unload` fusiona los pesos y devuelve el modelo base ya modificado.
merged_model = model_to_merge.merge_and_unload()
# --- Guardamos el modelo fusionado y el tokenizador ---
merged_model.save_pretrained(merged_model_path)
# También guarda el tokenizador en la misma carpeta para que sea un modelo autocontenido
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_model_path)

print("\n¡Modelo fusionado y guardado exitosamente!")
print(f"Ahora puedes cargar este modelo directamente desde '{merged_model_path}'")