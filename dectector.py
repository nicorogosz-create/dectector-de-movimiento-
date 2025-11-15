import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

umbral_area_minima = 500
actualizar_fondo_cada_segundos = 5
ultimo_actualizado = time.time()

# Leer primer frame para fondo inicial
ret, fondo = cap.read()
if not ret or fondo is None:
    print("Error: No se pudo capturar el primer frame.")
    cap.release()
    exit()

fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)
fondo = cv2.GaussianBlur(fondo, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: no se pudo leer frame de la cámara.")
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (21, 21), 0)

    # Actualizar fondo para adaptación
    if time.time() - ultimo_actualizado > actualizar_fondo_cada_segundos:
        fondo = gris
        ultimo_actualizado = time.time()

    diff = cv2.absdiff(fondo, gris)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Contornos de movimiento
    contornos_mov, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar rectángulos solo para movimiento
    for cm in contornos_mov:
        if cv2.contourArea(cm) < umbral_area_minima:
            continue
        x_m, y_m, w_m, h_m = cv2.boundingRect(cm)
        cv2.rectangle(frame, (x_m, y_m), (x_m+w_m, y_m+h_m), (0, 255, 0), 2)

    cv2.imshow("Solo Movimiento Detectado", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()