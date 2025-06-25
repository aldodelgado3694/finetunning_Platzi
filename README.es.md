# Fine-tunning a modelo gemma en local

Hola, en este repositorio me puse a la tarea de hacer fine-tuning al modelo de gemma-2b-it como se hace en el curso de platzi [Desarrollo de Chatbots con OpenAI](https://platzi.com/cursos/openai-api-23/). Se usa SFT (Supervised Fine Tunning) para entrenar el modelo. Además, para que este proceso sea factible en hardware con memoria limitada (como una GPU de laptop en mi caso), se emplea una técnica de optimización llamada LoRA (Low-Rank Adaptation), que es una forma de PEFT (Parameter-Efficient Fine-Tuning).

Cabe resaltar que este proyecto no es usable en producción pues detecte que el dataset que proporcionan el curso esta altamente contaminado con frases en Portugués y no tiene coherencia en el formato. Además que se optimiza el proceso lo mas posible para poder hacerlo en una computadora con recursos "limitados". Incluso pienso que teniendo un dataset decente ( Sin contenido basura o rico en ejemplos) no podría ser entrenado en una computadora con las capacidades en las que se probo. Sin embargo existen muchas formas poder tener poder computacional rentado como Google Colab y este código es un gran punto de entrada para poder especializar tu propio modelo.

### Objetivo 

- Hacer el curso sin necesidad de pagar por open AI
- Hacer pruebas un modelo en HigginFace
- Crear tu propio modelo modificado y hacer pruebas con el.

También siéntete libre de usar esto para jugar con un modelo en local


## Empecemos

#### Hardware

Yo tengo una laptop con una RTX3050 con 6 GB de VRAM, procesador 12th Gen Intel Core i5-12450HX  2.40 GHz de 8 núcleos, 16 GB de memoria RAM.

#### Requisitos

- De preferencia usar WSL2 si usas windows
- Si usas ya Linux o en WSL instalar Conda
- Tener instalado y actualizados los drivers de tu tarjeta grafica así como la librería de cuda.
- Tener una cuenta en HuggingFace y visitar los modelos que quieras ejecutar para poder pedir el permiso de uso.

### Instalación 

### Crear y activar el entorno con Conda (Usaremos python 3.11)

```sh
conda create --name qlora-env python=3.11 -y
conda activate qlora-env

```

### Accede a la carpeta del proyecto

```sh
cd finetunning_Platzi
```

### Instala las dependencias

```sh
pip3 install -r requirements.txt
```

### Crea un token y regístralo en tu equipo 

Para mas información [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

Para crear el token [Tokens de acceso de usuarios](https://huggingface.co/docs/hub/security-tokens)

```sh
pip3 install -U "huggingface_hub[cli]" 

huggingface-cli login

#Seguir los pasos e introducir el token
```

### Entendiendo el código y cambiando parámetros

Te invito a ver el código, no esta suficientemente comentado pero creo que se explica bien que hace cada cosa.

Lo archivos importantes son:

| Archivo | ¿Qué hace? |
|---------|------------|
| qlora.py | Es el programa principal para el entrenamiento, es importante modificar en el parámetro la carpeta de salida donde se cargara los adaptadores. [qlora.py linea 8]()|
| inferencia_directa.py | Este programa nos sirve para poder ejecutar por ejemplo el modelo de google gemma 2 antes de el entrenamiento, solo es necesario cambiar el nombre del modelo en [inferencia_directa.py linea 6](), el modelo ahora cargado es el que ya entrene y fusione|
| inferencia_adaptada.py | Este programa nos permite probar el modelo después del entrenamiento y antes de la fusión, en este casi también es necesario modificar el parámetro de la carpeta de salida y debe ser igual que el que se configuro en el archivo de qlora.py [inferencia_adaptada.py linea 6]()|
| fusionar_modelo.py | Este programa es el que nos ayudara a hacer la fusión de nuestro adaptador obtenido después del entrenamiento, al finalizar se creara la carpeta "gemma-2b-it-platzibot" la cual contendrá el modelo fusionado y es el que podremos subir o exportar a donde necesitemos para usar el modelo. También es necesario modificar el parámetro de carpeta de salida a igual que en qlora.py en [fusionar_modelo.py linea 7]()|


## Iniciando proyecto

Ya con esto solo tenemos que ejecutar el comando en nuestra terminal o editor de código.

No olvides antes modificar la carpeta de salida en [qlora.py linea 8]() o bien si quieres cambiar el modelo base lo puedes hacer en [qlora.py linea 17]()
Lee el código y te invito a cambiar parámetros como "num_train_epochs" para definir cuantas épocas entrenar, "per_device_train_batch_size" si es que tienes mejor hardware, o ayúdame a mejorar la eficiencia o la calidad del entrenamiento con parámetros que no haya incluido o este configurando mal.

- Ejecuta en terminal

```sh
python3 qlora.py
```

Al terminar de correr el entrenamiento, pues probar el resultado. De igual manera si cambiaste las carpeta en qlora.py hazlo también en [inferencia_adaptada.py linea 6]().

```sh
python3 inferencia_adaptada.py
```

Hasta este punto es mas que suficiente si solo quieres hacer pruebas, ya viste lo que es capaz de hacer el entrenamiento, y puedes usar el código de inferencia_adaptada.py para crear tu propio chat. Sin embargo aun hay mas.

Fusiona el modelo base con el adaptador para que tengas a la mano el modelo o lo puedas compartir fácilmente con quien desees. No olvide modificar la carpeta de salida en [fusionar_modelo.py linea 7]().

```sh
python3 fusionar_modelo.py
```

Ahora que ya esta listo, puedes subir tu modelo a HuggingFace y hacer uso de el con el archivo de "inferencia_directa.py" o también puedes usar este programa para probar el modelo antes o después de entrenar tu modelo así ver que tan capaz es el modelo base de responder a preguntas similares o probar que tan bien quedo entrenado tu modelo. Si solo quieres probar un modelo no olvides modificar el modelo base en [inferencia_directa.py linea 6]()

```sh
python3 inferencia_directa.py
```


# Aprende y ayúdame a aprender

Te invito a que me ayudes a detectar errores o malas practicas en este código. Si puedes ejecutar este código en algún equipo con menores capacidades, puedes ayudar compartiendo tus parámetros. Si tienes algún problema estaré mas que emocionado de ayudarte (Aun que por ahora me es difícil trabajar diario en algo, me tratare de dar el tiempo de ayudarte). Te agradezco que hayas llegado hasta aquí.