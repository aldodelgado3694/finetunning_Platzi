
import requests
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Delete this line if you dont want to use the .env file
# The .env file is used to store the token in a secure way
#--------------------------------------------------------
from dotenv import load_dotenv
import os
# Load the .env file
load_dotenv()
# Get the token from the .env file
TOKEN = os.getenv("TELEGRAM_TOKEN")
#--------------------------------------------------------

#Uncomment this lines if you want to use an Telegram Token
# And put your token
#TOKEN = "TELEGRAM_TOKEN"

def get_updates(offset):
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    response = requests.get(url, params=params)
    return response.json()["result"]

def send_messages(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": chat_id, "text": text}
    response = requests.post(url, params=params)
    return response

def prepare_model():
    base_model_name = "aldodelgado3694/gemma-2b-it-platzibot"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return tokenizer, model

def get_openai_response(pregunta, tokenizer, model):
    try:
        chat = [
            { "role": "user", "content": f"""Eres un asistente de atención a clientes y estudiantes de la plataforma de educación online en tecnología, inglés y liderazgo llamada Platzi.
        Necesitas responder a la siguiente pregunta: {pregunta}""" },
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            # Penaliza la repetición para evitar bucles.
            repetition_penalty=1.2,
        )

        respuesta_generada = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        last_punctuation = max(respuesta_generada.rfind('.'), respuesta_generada.rfind('!'), respuesta_generada.rfind('?'))
        if last_punctuation != -1:
            respuesta_limpia = respuesta_generada[:last_punctuation + 1]
        else:
            respuesta_limpia = respuesta_generada
        return respuesta_limpia
    except Exception:
        return "There was an error"


def main():
    print("Starting bot...")
    tokenizer, model = prepare_model()
    print("Bot is ready")
    offset = 0
    while True:
        if updates := get_updates(offset):
            for update in updates:
                offset = update["update_id"] +1
                chat_id = update["message"]["chat"]['id']
                user_message = update["message"]["text"]
                print(f"Received message: {user_message}")
                GPT = get_openai_response(user_message, tokenizer, model)
                send_messages(chat_id, GPT)
        else:
            time.sleep(1)


if __name__ == '__main__':
    main()