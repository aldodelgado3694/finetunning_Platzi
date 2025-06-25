# Fine-tuning a local Gemma model

Hello, in this repository I took on the task of fine-tuning the gemma-2b-it model as it's done in the Platzi course [Desarrollo de Chatbots con OpenAI](https://platzi.com/cursos/openai-api-23/). SFT (Supervised Fine-Tuning) is used to train the model. Additionally, to make this process feasible on hardware with limited memory (like a laptop GPU in my case), an optimization technique called LoRA (Low-Rank Adaptation) is employed, which is a form of PEFT (Parameter-Efficient Fine-Tuning).

It's worth noting that this project is not production-ready, as I detected that the dataset provided in the course is highly contaminated with Portuguese phrases and lacks format consistency. Furthermore, the process is optimized as much as possible to run on a computer with "limited" resources. I even think that with a decent dataset (without junk content or rich in examples), it couldn't be trained on a computer with the specs it was tested on. However, there are many ways to rent computing power, like Google Colab, and this code is a great starting point for specializing your own model.

### Objective

- Complete the course without needing to pay for OpenAI
- Test a model on Hugging Face
- Create your own modified model and test it.

Also, feel free to use this to play with a model locally.

## Let's get started

#### Hardware

I have a laptop with an RTX3050 with 6 GB of VRAM, a 12th Gen Intel Core i5-12450HX 2.40 GHz 8-core processor, and 16 GB of RAM.

#### Requirements

- Preferably use WSL2 if you're on Windows
- If you're already on Linux or in WSL, install Conda
- Have your graphics card drivers and the CUDA library installed and updated.
- Have a Hugging Face account and visit the models you want to run to request usage permission.

### Installation

### Create and activate the Conda environment (We'll use python 3.11)

```sh
conda create --name qlora-env python=3.11 -y
conda activate qlora-env
```

### Access the project folder

```sh
cd finetunning_Platzi
```

### Install the dependencies

```sh
pip3 install -r requirements.txt
```

### Create a token and register it on your machine

For more information [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

To create the token [User access tokens](https://huggingface.co/docs/hub/security-tokens)

```sh
pip3 install -U "huggingface_hub[cli]"

huggingface-cli login

#Follow the steps and enter the token
```

### Understanding the code and changing parameters

I invite you to look at the code. It's not heavily commented, but I believe it's self-explanatory.

The important files are:

| File | What it does? |
|---------|------------|
| qlora.py | This is the main program for training. It's important to modify the output directory parameter where the adapters will be loaded. [qlora.py line 8]()|
| inferencia_directa.py | This program allows us to run, for example, the Google Gemma 2 model before training. You just need to change the model name in [inferencia_directa.py line 6](). The model currently loaded is the one I already trained and merged. |
| inferencia_adaptada.py | This program lets us test the model after training and before merging. In this case, it's also necessary to modify the output directory parameter to be the same as the one configured in the qlora.py file. [inferencia_adaptada.py line 6]()|
| fusionar_modelo.py | This program helps us merge our adapter obtained after training. Upon completion, the "gemma-2b-it-platzibot" folder will be created, containing the merged model, which we can then upload or export wherever we need to use it. It's also necessary to modify the output directory parameter to match the one in qlora.py at [fusionar_modelo.py line 7](). |

## Starting the project

With that done, we just need to execute the command in our terminal or code editor.

Don't forget to first modify the output directory in [qlora.py line 8]() or, if you want to change the base model, you can do so in [qlora.py line 17](). Read the code and I invite you to change parameters like "num_train_epochs" to define how many epochs to train, "per_device_train_batch_size" if you have better hardware, or help me improve the efficiency or quality of the training with parameters I haven't included or have configured incorrectly.

- Run in terminal

```sh
python3 qlora.py
```

After the training finishes, you can test the result. Similarly, if you changed the directory in qlora.py, do the same in [inferencia_adaptada.py line 6]().

```sh
python3 inferencia_adaptada.py
```

Up to this point, this is more than enough if you just want to run tests. You've seen what the training is capable of, and you can use the inferencia_adaptada.py code to create your own chat. However, there's still more.

Merge the base model with the adapter so you have the model handy or can easily share it with whomever you wish. Don't forget to modify the output directory in [fusionar_modelo.py line 7]().

```sh
python3 fusionar_modelo.py
```

Now that it's ready, you can upload your model to Hugging Face and use it with the "inferencia_directa.py" file. You can also use this program to test the model before or after training to see how capable the base model is of answering similar questions, or to test how well your model was trained. If you just want to test a model, don't forget to modify the base model in [inferencia_directa.py line 6]().

```sh
python3 inferencia_directa.py
```

# Learn and help me learn

I invite you to help me detect errors or bad practices in this code. If you can run this code on a machine with lower specs, you can help by sharing your parameters. If you have any problems, I will be more than happy to help you (although for now it's difficult for me to work on this daily, I will try to make time to help you). Thank you for reading this far.