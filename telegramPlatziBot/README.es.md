# Proyecto de Curso de Desarrollo de Chatbots con OpenAI

Hola Platzinautas, en lugar de poner el servicio en un servidor, como se fomenta en el curso, mejor ajuste el código para usar todo lo que cree en este repositorio.

Ya que el curso se basa en hacer fine-tuning a un modelo de Open AI y luego usar este modelo en un Bot de Telegram, yo uso el modelo al que le hice el fine-tuning para que responda en el Bot de Telegram. 

## Par ejecutarlo solo ejecutamos `platzi_bot.py`

Debemos de seguir los pasos en el README.md del proyecto inicial

Crear tu Bot y obtener el token[Crear Bot en telegram](https://core.telegram.org/bots/tutorial)

Crear en la carpeta raíz del proyecto un el archivo `.env` y agrega la siguiente linea, sustituyendo `YOUR_TELEGRAM_TOKEN` por tu token.

```python
TELEGRAM_TOKEN = YOUR_TELEGRAM_TOKEN
```

Luego entrar a la carpeta del ejemplo:

```sh
cd telegramPlatziBot
```

Y ejecutar el script:

```sh
python3 platzi_bot.py
```

Debes ver algo como:

```sh
Starting bot...
Loading checkpoint shards: 100%|████████████████| 2/2 [00:07<00:00,  3.74s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
Bot is ready
Received message: Que necesito para aprender JavaScript
```

## Prueba y disfruta

Ojala que te sea de utilidad este ejemplo, espero subir mas próximamente.