import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTTrainer

#Variables de entorno
output_dir = "results-with-eval-20epochs"

#Carga y preparación del Dataset ---

train_dataset = load_dataset("aldodelgado3694/data_train", split="train")
eval_dataset = load_dataset("aldodelgado3694/data_val", split="validation")


# Configuración y carga del modelo base ---
model_name = "google/gemma-2b-it"

# Carga el modelo directamente en la GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="QUESTION_ANS", #CAUSAL_LM
)

# --- Argumentos para un Entrenamiento ---
training_args = TrainingArguments(
    # Directorio de salida
    output_dir = f"./{output_dir}",

    # --- Parámetros de Entrenamiento ---
    num_train_epochs=20,  # Número de veces que se recorrerá todo el dataset. 3 es un buen valor inicial.

    # --- Gestión de Memoria y Batch Size ---
    per_device_train_batch_size=2,  # Puedes intentar subirlo a 2 o 4 si tienes suficiente VRAM.
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # En 8 para intentar compensar el batch size bajo .

    # --- Optimizador y Learning Rate ---
    learning_rate=2e-4,
    optim="paged_adamw_8bit", # Un optimizador que ahorra memoria, ideal para QLoRA.

    # --- Checkpoints y Logging ---
    logging_steps=25,               # Muestra el loss cada 25 pasos.
    eval_strategy="steps",      # Evaluar el modelo cada X pasos
    eval_steps=75,                    # Frecuencia de la evaluación
    save_strategy="steps",            # Guardar checkpoints según la misma frecuencia
    save_steps=58,
    load_best_model_at_end=True,      # Cargar el mejor checkpoint al final del entrenamiento
    metric_for_best_model="eval_loss",# Métrica para decidir cuál es el "mejor" modelo
    greater_is_better=False,         # Para 'loss', un valor más bajo es mejor

    # --- ¡CLAVE! Desactiva gradientes durante la evaluación ---
    prediction_loss_only=True, # Solo calcula el loss, no los logits completos
    eval_on_start = True,

    # --- Configuración de Precisión ---
    fp16=True, # Indispensable para un entrenamiento rápido en GPUs NVIDIA.
)

#Tranforma el dataset del Curso de Desarrollo de Chatbots con OpenAI a uno que pueda entender el modelo de gemma
def transformar_conversacion(ejemplo):
    # Tu función está perfecta
    mensajes = ejemplo["messages"]
    mensajes_transformados = []
    system_prompt = ""
    for mensaje in mensajes:
        if mensaje["role"] == "system":
            system_prompt = mensaje["content"]
            break
    for i, mensaje in mensajes:
        if mensaje["role"] == "user":
            if system_prompt:
                contenido_combinado = f"{system_prompt}\n\n{mensaje['content']}"
                mensajes_transformados.append({"role": "user", "content": contenido_combinado})
                system_prompt = ""
            else:
                mensajes_transformados.append(mensaje)
        elif mensaje["role"] != "system":
            mensajes_transformados.append(mensaje)
    return {"messages": mensajes_transformados}

#Convertimos los dataset
train_dataset = train_dataset.map(transformar_conversacion)
eval_dataset = eval_dataset.map(transformar_conversacion)


# Inicialización y Entrenamiento ---

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,    # <-- Pasamos el dataset de entrenamiento
    eval_dataset=eval_dataset,       # <-- Pasamos el dataset de evaluación
    peft_config=lora_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], 
    # <-- Añadimos el callback para detener el entrenamiento si se empieza a volver malo
)

# Entrenando modelo ---
trainer.train()

print("\n¡¡¡ENTRENAMIENTO COMPLETADO!!! ¡FELICIDADES!")

# Guardando el mejor modelo ---
best_model_path = f"./{output_dir}/best_model"
trainer.save_model(best_model_path)

print(f"El mejor modelo ha sido guardado en: {best_model_path}")