# convolucion_mpi.py
from mpi4py import MPI
import numpy as np
import cv2
import sys

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

def aplicar_convolucion_mpi(seccion, kernel):
    k_altura, k_ancho = kernel.shape
    altura, ancho = seccion.shape
    resultado = np.zeros((altura - k_altura + 1, ancho - k_ancho + 1))

    for i in range(altura - k_altura + 1):
        for j in range(ancho - k_ancho + 1):
            submatriz = seccion[i:i+k_altura, j:j+k_ancho]
            resultado[i, j] = np.sum(submatriz * kernel)

    return resultado

def main():
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # El proceso principal carga la imagen y elige el kernel
        imagen_path = sys.argv[1]
        kernel_name = sys.argv[2]

        # Cargar la imagen en escala de grises
        imagen_gris = image_to_grayscale_matrix(imagen_path)

        # Seleccionar el kernel
        # Aquí puedes agregar tus propios kernels o una lógica para seleccionarlos
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
                ])
        }
        kernel = kernels.get(kernel_name, kernels["kernel_laplace"])

        # Dividir la imagen para distribuir a los procesos
        secciones = np.array_split(imagen_gris, size, axis=0)
    else:
        secciones = None
        kernel = None

    # Distribuir secciones y kernel a cada proceso
    seccion = comm.scatter(secciones, root=0)
    kernel = comm.bcast(kernel, root=0)

    # Cada proceso aplica la convolución a su sección
    resultado_seccion = aplicar_convolucion_mpi(seccion, kernel)

    # Recolectar los resultados en el proceso principal
    resultados = comm.gather(resultado_seccion, root=0)

    if rank == 0:
        # Combinar los resultados
        resultado_final = np.concatenate(resultados, axis=0)
        # Guardar el resultado en un archivo .npy
        np.save('resultado_convolucion.npy', resultado_final)

if __name__ == "__main__":
    main()
