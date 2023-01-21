Detección de Naranjas en buenas y malas
======

# Dataset

## Naranjas Buenas

![Naranjas Buenas](/imagenes/naranjas-buenas.png)

## Naranjas Malas

![Naranjas Malas](/imagenes/naranjas-malas.png)




# Version de Python

Python 3.9

# Instalación del entorno

```bash

virtualenv --python=python3.10 .venv

source venv/bin/activate

pip install -r requirements.txt

```

[PyTorch](https://www.gcptutorials.com/post/how-to-install-pytorch-with-pip)


# Recorte y preparacion de imagenes

## Recorte
Las fotografias tomadas se recortaron usando la aplicacion gThumb

![Recorte de fotos](/imagenes/recortar-fotos.jpeg)

# Redimensionamiento de imagenes

Para redimensionar a un tamaño de 500x500 se usó el utilitario convert de linea de comandos.

```bash

find . -maxdepth 1 -iname "*.jpg" | xargs -L1 -I{} convert -resize 500x500 "{}" resize/"{}"

```


# Entrenamiento

```bash

python training

```

[Logs](logs-training.md)


![Valid Loss and Train Loss](/imagenes/ValidLossandTrainLoss.png)


![Valid Loss and Valid Accuracy](/imagenes/ValidLossandValidAccuracy.png)


# Pruebas

```bash

python testing

```


![Test 1](/imagenes/test1.png)

![Test 2](/imagenes/test2.png)

![Test 3](/imagenes/test3.png)

![Test 4](/imagenes/test4.png)

![Test 5](/imagenes/test5.png)

![Test 6](/imagenes/test6.png)

![Test 7](/imagenes/test7.png)

![Test 8](/imagenes/test8.png)

![Test 9](/imagenes/test9.png)

![Test 10](/imagenes/test10.png)

![Test 11](/imagenes/test11.png)


# Camara

```bash

python pyqt6cam

```


# Basado en

https://iqraanwar.medium.com/how-to-detect-rotten-fruits-using-image-processing-python-be2d39abc709

https://github.com/IqraBaluch/Detection-of-Rotten-Fruits-DRF-Using-Image-Processing-Python


# Automatic classification of oranges using image processing and data mining techniques (Argentina)
https://core.ac.uk/download/pdf/296347784.pdf

# An extensive dataset for successful recognition of fresh and rotten fruits

https://www.sciencedirect.com/science/article/pii/S2352340922007594

# Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality (DATASET)

https://data.mendeley.com/datasets/bdd69gyhv8/1

https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/ccd1f142-03b2-473a-8c78-78920e63b8bd/file_downloaded