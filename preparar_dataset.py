import os
import random

from utils import *

def my_shuffle(lista, repeticiones=REPETICIONES_DESORDENAR):
    i = 0
    while i < repeticiones:
        random.shuffle(lista)
        i += 1


os.system('rm -rf %s' % RUTA_DESTINO)
# os.system('mkdir %s' % RUTA_DESTINO)
os.system('mkdir -p %s/train/%s' % (RUTA_DESTINO, NARANJAS_FRESCA))
os.system('mkdir -p %s/train/%s' % (RUTA_DESTINO, NARANJAS_NOFRESCA))
os.system('mkdir -p %s/valid/%s' % (RUTA_DESTINO, NARANJAS_FRESCA))
os.system('mkdir -p %s/valid/%s' % (RUTA_DESTINO, NARANJAS_NOFRESCA))
os.system('mkdir -p %s/test/%s' % (RUTA_DESTINO, NARANJAS_FRESCA))
os.system('mkdir -p %s/test/%s' % (RUTA_DESTINO, NARANJAS_NOFRESCA))

for (ruta_fuente, clase) in (
        (RUTA_FUENTE_NARANJAS_FRESCAS, NARANJAS_FRESCA),
        (RUTA_FUENTE_NARANJAS_NOFRESCAS, NARANJAS_NOFRESCA)):

    lista_archivos = os.listdir(ruta_fuente)[:50]
    my_shuffle(lista_archivos)
    lista_train = lista_archivos[:int(TRAIN * len(lista_archivos))]
    lista_test = lista_archivos[-int(TEST * len(lista_archivos)):]
    lista_valid = list(set(lista_archivos) - set(lista_train) - set(lista_test))

    params = [
        (RUTA_DESTINO_TRAIN, lista_train),
        (RUTA_DESTINO_VALID, lista_test),
        (RUTA_DESTINO_TEST, lista_valid),
    ]

    for ruta_destino, lista_archivo in params:
        for archivo in lista_archivo:
            cmd = "cp '%s/%s' '%s/%s/%s'" % (ruta_fuente, archivo, ruta_destino, clase, archivo)
            os.system(cmd)
