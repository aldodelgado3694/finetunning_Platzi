# Chatbot Development Course Project with OpenAI

Hello Platzinautas, instead of deploying the service on a server, as encouraged in the course, I've adjusted the code to use everything created within this repository.

Since the course focuses on fine-tuning an OpenAI model and then using this model in a Telegram Bot, I use the fine-tuned model directly to respond within the Telegram Bot.

## How to Run: Execute `platzi_bot.py`

You should follow the steps outlined in the original project's README.md:

Create your Bot and get the token: [Create a Telegram Bot](https://core.telegram.org/bots/tutorial)

Create a `.env` file in the project's root folder and add the following line, replacing `YOUR_TELEGRAM_TOKEN` with your token:

```python
TELEGRAM_TOKEN = YOUR_TELEGRAM_TOKEN
```

Next, navigate to the example folder:

```sh
cd telegramPlatziBot
```

And execute the script:

```sh
python3 platzi_bot.py
```

You should see output similar to:

```sh
Starting bot...
Loading checkpoint shards: 100%|████████████████| 2/2 [00:07<00:00,  3.74s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
Bot is ready
Received message: Que necesito para aprender JavaScript
```

## Test and Enjoy!

Hopefully, you find this example useful. I aim to upload more soon.
