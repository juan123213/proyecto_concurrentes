import streamlit as st
import numpy as np
import threading
import subprocess
from bing_image_downloader import downloader
import cv2
import random
import multiprocessing

def convolucion_paralela(imagen_gris, kernel, num_procesos):
    altura, ancho = imagen_gris.shape
    k_altura, k_ancho = kernel.shape

    # Determinar el tamaño de la sección de la imagen que cada proceso manejará
    seccion_altura = altura // num_procesos

    # Dividir la imagen en secciones para cada proceso
    secciones = [imagen_gris[i*seccion_altura:(i+1)*seccion_altura, :] for i in range(num_procesos)]

    # Crear un proceso para cada sección
    procesos = []
    manager = multiprocessing.Manager()
    resultados = manager.list([None]*num_procesos)  # Almacenará los resultados de cada proceso

    for i in range(num_procesos):
        proceso = multiprocessing.Process(target=aplicar_convolucion, args=(secciones[i], kernel, resultados, i))
        procesos.append(proceso)
        proceso.start()

    # Esperar a que todos los procesos terminen
    for proceso in procesos:
        proceso.join()

    # Combinar los resultados
    resultado_final = np.concatenate(resultados)
    return resultado_final

def aplicar_convolucion(seccion, kernel, resultados, indice):
    k_altura, k_ancho = kernel.shape
    altura, ancho = seccion.shape
    resultado = np.zeros((altura - k_altura + 1, ancho - k_ancho + 1))

    for i in range(altura - k_altura + 1):
        for j in range(ancho - k_ancho + 1):
            submatriz = seccion[i:i+k_altura, j:j+k_ancho]
            resultado[i, j] = np.sum(submatriz * kernel)

    resultados[indice] = resultado



def image_to_grayscale_matrix(image_path):
    # Leer la imagen
    original_image = cv2.imread(image_path)

    # Verificar si la lectura de la imagen fue exitosa
    if original_image is None:
        print(f"No se pudo leer la imagen en {image_path}")
        return None

    # Convertir la imagen a escala de grises
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Obtener la matriz de la imagen en escala de grises
    grayscale_matrix = np.array(grayscale_image)

    return grayscale_matrix

def convolucionsecuencial(imagen, kernel):
    altura, ancho = imagen.shape
    k_altura, k_ancho = kernel.shape
    resultado = np.zeros((altura - k_altura + 1, ancho - k_ancho + 1))

    for i in range(altura - k_altura + 1):
        for j in range(ancho - k_ancho + 1):
            submatriz = imagen[i:i+k_altura, j:j+k_ancho]
            resultado[i, j] = np.sum(submatriz * kernel)

    return resultado



def download_images(tema, numerohilo):
    downloader.download(f'{tema} {numerohilo}', limit=100,  output_dir='downloads', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)


def descarga():
    #Dirección de carpeta para cargar imagenes
    downloads = "downloads"

    # Ejecutar los hilos
    hilo1.start()
    hilo2.start()
    hilo3.start()
    hilo4.start()
    hilo5.start()
    hilo6.start()
    hilo7.start()
    hilo8.start()
    hilo9.start()
    hilo10.start()

    # Esperar a que terminen los hilos ejecutados
    hilo1.join()
    hilo2.join()
    hilo3.join()
    hilo4.join()
    hilo5.join()
    hilo6.join()
    hilo7.join()
    hilo8.join()
    hilo9.join()
    hilo10.join()

def obtener_10_imagenes_aleatorias(base_path, num_images=10):
    imagenes = []

    for _ in range(num_images):
        # Generar un número aleatorio entre 1 y 99
        num = random.randint(1, 10)
        # Construir el nombre del archivo de la imagen
        nombre_imagen = f"Image_{num}.jpg"
        # Construir la ruta completa de la imagen
        ruta_imagen = f"/downloads/{num}/{nombre_imagen}"
        # Añadir la ruta a la lista de imágenes
        imagenes.append(ruta_imagen)

    return imagenes




#Kernels

kernels= {
  "kernel_class_1" : np.array([
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, -1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
  ]),
  "kernel_class_2" : np.array([
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, -1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
  ]),
  "kernel_class_3" : np.array([
      [0, 0, -1, 0, 0],
      [0, 0, 3, 0, 0],
      [0, 0, -3, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0]
  ]),
  "kernel3x3" : np.array([
      [0, 0, 0, 0, 0],
      [0, -1, 2, -1, 0],
      [0, 2, -4, 2, 0],
      [0, -1, 2, -1, 0],
      [0, 0, 0, 0, 0]
  ]),
  "edge3x3" : np.array([
      [0, 0, 0, 0, 0],
      [0, -1, 2, -1, 0],
      [0, 2, -4, 2, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
  ]),
  "square5x5" : np.array([
      [-1,2, -2, 2, -1],
      [2,-6, 8, -6,  2],
      [-2, 8,-12, 8,-2],
      [2, -6, 8, -6, 2],
      [-1, 2, -2, 2, -1]
  ]),
  "edge5x5" : np.array([
      [-1,2, -2, 2, -1],
      [2,-6, 8, -6,  2],
      [-2, 8,-12, 8,-2],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, -0]
  ]),

  "kernelsobelv" : np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ]),
  "kernelsobelh" : np.array([
      [-1, 2, -1],
      [0, 0, 0],
      [1, 2, 1]
  ]),
  "kernel_laplace": np.array([
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1]
  ]),
  "kernelprewittv": np.array([
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1]
  ]),
  "kernelprewitth" : np.array([
      [-1, -1,-1],
      [0, 0, 0],
      [1, 1, 1]
    ])}
st.title("Proyecto Final - Programación Concurrente y Distribuida")

tema_descargas = st.text_input("Y hoy ¿Acerca de qué quieres descargar imágenes?")

if st.button("Descargar imágenes con la configuración seleccionada", key="button1"):
    if tema_descargas == "":
        st.error("No has seleccionado un tema de descargas")
        #print("hilos ",hilos)
        #print("framework ",framework)
        #print("tema_descargas",tema_descargas)
        #print("kernel",kernel)

    else:

        st.success("Descargando imágenes de {}".format(tema_descargas))
        # Crear los hilos
        hilo1 = threading.Thread(target=download_images, name='Hilo 1', args=(tema_descargas,1,))
        hilo2 = threading.Thread(target=download_images, name='Hilo 2', args=(tema_descargas,2,))
        hilo3 = threading.Thread(target=download_images, name='Hilo 3', args=(tema_descargas,3,))
        hilo4 = threading.Thread(target=download_images, name='Hilo 4', args=(tema_descargas,4,))
        hilo5 = threading.Thread(target=download_images, name='Hilo 5', args=(tema_descargas,5,))
        hilo6 = threading.Thread(target=download_images, name='Hilo 6', args=(tema_descargas,6,))
        hilo7 = threading.Thread(target=download_images, name='Hilo 7', args=(tema_descargas,7,))
        hilo8 = threading.Thread(target=download_images, name='Hilo 8', args=(tema_descargas,8,))
        hilo9 = threading.Thread(target=download_images, name='Hilo 9', args=(tema_descargas,9,))
        hilo10 = threading.Thread(target=download_images, name='Hilo 10', args=(tema_descargas,10,))
        descarga()

st.button("mostrar 10 imágenes random de las descargadas", key="button3")

framework= st.radio("¿Qué tipo de framework o librería quieres usar?", ("C", "OpenMP", "MPI4py", "PyCUDA", "multiprocessing"))
st.success(f"Librería seleccionado: {framework}")

kernel_names = list(kernels.keys())

selected_kernel_name = st.radio("Selecciona un Kernel", kernel_names)

selected_kernel = kernels[selected_kernel_name]
st.success(f"Kernel seleccionado: {selected_kernel_name}")

hilos= st.radio ("¿cantidad de hilos quieres usar?", ("1", "2", "4", "6", "8"))


if st.button("Aplicar filtro a las imágenes", key="button3"):
    if (framework == "multiprocessing"):
        convolucion_paralela()

    else:

       "hello"
