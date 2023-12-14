import streamlit as st
import numpy as np
import threading
import subprocess
from bing_image_downloader import downloader
import cv2
import random
import multiprocessing
import sys
import os
import time


def convolucion_paralela_multiprocessing(imagen_gris, kernel, num_procesos):
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

    # Aplicar ecualización del histograma para mejorar el contraste
    contrasted_image = cv2.equalizeHist(grayscale_image)

    # Obtener la matriz de la imagen con mayor contraste
    contrasted_matrix = np.array(contrasted_image)

    return contrasted_matrix

def matriz_a_imagen_gris(matriz):
    # Normaliza la matriz para asegurar que los valores estén en el rango 0-255
    matriz_normalizada = ((matriz - np.min(matriz)) / (np.max(matriz) - np.min(matriz)) * 255).astype(np.uint8)

    # Convierte la matriz normalizada a una imagen en escala de grises
    imagen_gris = cv2.cvtColor(matriz_normalizada, cv2.COLOR_GRAY2BGR)

    return imagen_gris

def convolucionsecuencial(imagen, kernel):
    
    altura, ancho = imagen.shape
    k_altura, k_ancho = kernel.shape
    resultado = np.zeros((altura - k_altura + 1, ancho - k_ancho + 1))

    for i in range(altura - k_altura + 1):
        for j in range(ancho - k_ancho + 1):
            submatriz = imagen[i:i+k_altura, j:j+k_ancho]
            resultado[i, j] = np.sum(submatriz * kernel)
    end_time = time.time()

    return resultado 



def download_images(tema, numerohilo):
    # Descargar imágenes
    downloader.download(f'{tema} {numerohilo}', limit=20, output_dir='downloads', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    
    # Ruta del directorio de descargas
    download_dir = f'downloads/{tema} {numerohilo}/'
    
    # Filtrar y mover solo las imágenes .jpg al directorio de descargas principal
    for filename in os.listdir(download_dir):
        if filename.lower().endswith('.jpg'):
            source_path = os.path.join(download_dir, filename)
            destination_path = os.path.join('downloads', filename)
            os.rename(source_path, destination_path)     
            
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

#funcion para mostrar 10 imagenes aleatorias de la carpeta descargas

def show_random_images(tema):
    # Obtener la lista de imágenes en la carpeta
    image_names = os.listdir(f"downloads/{tema} 2")

    # Obtener 10 imágenes aleatorias
    random_image_names = random.sample(image_names, 5)

    # Mostrar las imágenes
    for image_name in random_image_names:
        #imagen=image_to_grayscale_matrix("downloads/super heroes 1/Image_2.jpg")
        image_path = os.path.join(f"downloads/{tema} 1/", image_name)
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(image, caption=image_name, use_column_width=True)
        show_stats(image)



def show_secuencial_images(tema,kernel):
    
    ruta=f"downloads/{tema} 1/"
    
    start_time = time.time()
    for iter in range(1,11):
        imagen=image_to_grayscale_matrix(f"{ruta}Image_{iter}.jpg")
        print(imagen)
        resultado=convolucionsecuencial(imagen,kernel)
        resultado= matriz_a_imagen_gris(resultado)
        st.image(resultado, caption="image_name", use_column_width=True)
        show_stats(resultado)
    tiempoSecu = end_time - start_time
    st.write("Tiempo de ejecución secuencial: ", tiempoSecu)


# Función para mostrar las estadísticas de una imagen
def show_stats(image):
    st.write("Shape:", image.shape)
    st.write("DType:", image.dtype)
    st.write("Min. value:", image.min())
    st.write("Max value:", image.max())
    st.write("Mean:", image.mean())



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

framework= st.radio("¿Qué tipo de framework o librería quieres usar?", ("secuencial","C", "OpenMP", "MPI4py", "PyCUDA", "multiprocessing"))
st.success(f"Librería seleccionado: {framework}")

kernel_names = list(kernels.keys())
selected_kernel_name = st.radio("Selecciona un Kernel", kernel_names)
selected_kernel = kernels[selected_kernel_name]
st.success(f"Kernel seleccionado: {selected_kernel_name}")

hilos= st.radio ("¿cantidad de hilos quieres usar?", ("1", "2", "4", "6", "8"))
st.success(f"hilos seleccionado: {hilos}")

procesos= st.radio ("¿cantidad de procesos que quieres usar?", ("1", "2", "4", "6", "8"))
st.success(f"procesos seleccionado: {procesos}")

if st.button("Aplicar filtro a las imágenes", key="button5"):
    imagen=image_to_grayscale_matrix(f"downloads/{tema_descargas} 1/Image_1.jpg")

    if (framework == "secuencial"):
        
        show_secuencial_images(tema_descargas,selected_kernel)
        #resultado=convolucionsecuencial(imagen,selected_kernel)
        #resultado= matriz_a_imagen_gris(resultado)
        #st.image(resultado, caption='Descripción de la imagen', use_column_width=True)
        
    elif (framework == "multiprocessing"):
        resultado=convolucion_paralela_multiprocessing(imagen,selected_kernel,int(procesos))
        resultado= matriz_a_imagen_gris(resultado)
        st.image(resultado, caption='Descripción de la imagen', use_column_width=True)       

    elif (framework == "MPI4py"):
        comando = ['mpiexec', '-n', procesos, sys.executable, 'convolucionmpi4.py', f"downloads/{tema_descargas} 1/Image_1.jpg", selected_kernel_name]
        procesompi = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Leer la salida y error del subproceso
        salida, error = procesompi.communicate()

        if procesompi.returncode == 0:
            st.success("Convolución completada con éxito.")
            
            # Cargar el resultado de la matriz
            try:
                resultado_convolucion = np.load('resultado_convolucion.npy')
                # Aquí puedes hacer algo con la matriz, como mostrarla
                st.image(resultado_convolucion, caption='Descripción de la imagen', use_column_width=True)
            except IOError:
                st.error("No se pudo cargar el resultado de la convolución.")
        else:
            st.error("Error en la ejecución de la convolución MPI.")
            st.text(error.decode())
            
    if (framework == "OpenMP"):
        # Compilar el programa en C
        subprocess.run(["gcc", "convolucion_openmp", "convolucion_openmp.c", "-lpthread"])

        # Ejecutar el programa compilado
        proceso = subprocess.Popen(["./convolucion_openmp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Leer la salida del subproceso
        salida, error = proceso.communicate()

        if proceso.returncode == 0:
            st.success("Convolución completada con éxito.")
            # Procesar y mostrar la salida
            st.text(salida.decode())
        else:
            st.error("Error en la ejecución de la convolución en C.")
            st.text(error.decode())
    
    if(tema_descargas!=""):
        st.success("Mostrando 10 imágenes aleatorias de la carpeta descargas")
        show_random_images(tema_descargas)
    
    
    else:

       st.error("error")
