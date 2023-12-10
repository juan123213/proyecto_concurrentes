import threading        # Manejo de hilos
import subprocess
from bing_image_downloader import downloader
import cv2
import numpy as np
import streamlit as st


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



def download_images(x):
    downloader.download(x, limit=100,  output_dir='downloads', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)


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


# Crear los hilos
hilo1 = threading.Thread(target=download_images, name='Hilo 1', args=("mazda car",))
hilo2 = threading.Thread(target=download_images, name='Hilo 2', args=("audi car",))
hilo3 = threading.Thread(target=download_images, name='Hilo 3', args=("ferrari car",))
hilo4 = threading.Thread(target=download_images, name='Hilo 4', args=("renault car",))
hilo5 = threading.Thread(target=download_images, name='Hilo 5', args=("gallardo car",))
hilo6 = threading.Thread(target=download_images, name='Hilo 6', args=("aventador car",))
hilo7 = threading.Thread(target=download_images, name='Hilo 7', args=("vyper car",))
hilo8 = threading.Thread(target=download_images, name='Hilo 8', args=("camaro car",))
hilo9 = threading.Thread(target=download_images, name='Hilo 9', args=("veneno car",))
hilo10 = threading.Thread(target=download_images, name='Hilo 10', args=("mclaren car",))




# Ruta de la imagen
image_path = "downloads/ferrari car/Image_39.jpg"

# Obtener la matriz de la imagen en escala de grises
grayscale_matrix = image_to_grayscale_matrix(image_path)


#Kernels

kernel_class_1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
kernel_class_2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
kernel_class_3 = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
])
kernel3x3 = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0]
])
edge3x3 = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
square5x5 = np.array([
    [-1,2, -2, 2, -1],
    [2,-6, 8, -6,  2],
    [-2, 8,-12, 8,-2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
])
edge5x5 = np.array([
    [-1,2, -2, 2, -1],
    [2,-6, 8, -6,  2],
    [-2, 8,-12, 8,-2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, -0]
])

kernelsobelv = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
kernelsobelh = np.array([
    [-1, 2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
kernel_laplace= np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])
kernelprewittv = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
kernelprewitth = np.array([
    [-1, -1,-1],
    [0, 0, 0],
    [1, 1, 1]
])






# Definir el kernel como una matriz de NumPy
kernelprueba = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])






st.title("""Proyecto Final - Programación Concurrente y Distribuida""")

tema_descargas = st.text_input("Y hoy ¿Acerca de qué quieres descargar imágenes?") 

if st.button("Descargar imágenes con la configuración seleccionada"):
    if tema_descargas == "":
        st.error("No has seleccionado un tema de descargas")
        #print("hilos ",hilos)
        #print("framework ",framework)
        #print("tema_descargas",tema_descargas)
        #print("kernel",kernel)
        
    else:
        seccion = st.e
        st.success("Descargando imágenes de {}".format(tema_descargas)) 
        
       
        
        st.button("mostrar 10 imagenes random de las descargadas")
            
        framework= st.radio("¿Qué tipo de framework o librería quieres usar?", ("C", "OpenMP", "MPI4py", "PyCUDA", "multiprocessing"))  


        kernel =st.radio("Qué tipo de filtro quieres usar?", ("El primero de los Class 1", "El primero de los Class 2",
                                                        "El primero de los Class 3", "Square 3x3",
                                                        "El primero de los Edge 3x3", "Square 5x5",
                                                        "El primero de los Edge 5x5", " sobel vertical y horizontalmente",
                                                        "Laplace", "prewitt vertical y horizontalmente"))  


        hilos= st.radio ("¿cantidad de hilos quieres usar?", ("1", "2", "4", "6", "8"))  
        



        
        




