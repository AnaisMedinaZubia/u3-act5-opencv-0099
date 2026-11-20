## Medina Anais NC 0099   Fecha 20/11/2026

import cv2
import numpy as np

# 1. Cargar imagen
imgcarro_rojo = cv2.imread('carrorojo.jpg')

if imgcarro_rojo is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el nombre del archivo.")
    exit()

# 2. Conversión de espacio de color (BGR a Escala de Grises y HSV)
gray_image = cv2.cvtColor(imgcarro_rojo, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(imgcarro_rojo, cv2.COLOR_BGR2HSV)

# 3. Redimensionar imagen
resized_image = cv2.resize(imgcarro_rojo, (400, 300))

# 4. Recortar imagen (Slicing)
cropped_image = imgcarro_rojo[50:250, 50:250]

# 5. Rotar imagen (90 grados en sentido horario)
rotated_image = cv2.rotate(imgcarro_rojo, cv2.ROTATE_90_CLOCKWISE)

# 6. Desenfoque (Gaussian Blur)
blurred_image = cv2.GaussianBlur(imgcarro_rojo, (7, 7), 0)

# 7. Detección de bordes (Canny)
edges = cv2.Canny(gray_image, 100, 200)

# 8. Umbralización (Thresholding)
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 9. Operaciones morfológicas (Erosión y Dilatación)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(thresh_image, kernel, iterations=1)
dilation = cv2.dilate(thresh_image, kernel, iterations=1)

# 10. Detección de contornos
contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = imgcarro_rojo.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# 11. Dibujo de formas geométricas y texto
drawing_image = imgcarro_rojo.copy()
cv2.rectangle(drawing_image, (30, 30), (180, 180), (255, 0, 0), 2)
cv2.circle(drawing_image, (105, 105), 50, (0, 0, 255), -1)
cv2.putText(drawing_image, 'OpenCV', (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 12. Visualización de resultados
cv2.imshow('Imagen Original', imgcarro_rojo)
cv2.imshow('Escala de Grises', gray_image)
cv2.imshow('HSV', hsv_image)
cv2.imshow('Redimensionada', resized_image)
cv2.imshow('Recortada', cropped_image)
cv2.imshow('Rotada', rotated_image)
cv2.imshow('Desenfoque Gaussiano', blurred_image)
cv2.imshow('Bordes Canny', edges)
cv2.imshow('Umbralizacion', thresh_image)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilatacion', dilation)
cv2.imshow('Contornos', contour_image)
cv2.imshow('Dibujo y Texto', drawing_image)

# Guardar una imagen procesada como ejemplo
cv2.imwrite('resultado_bordes.jpg', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Medina Anais NC 0099  Fecha 20/11/2026")