# Proyecto personal: Nowcasting sobre imágenes de radar

En este repositorio se aborda la resolución de un problema de nowcasting utilizando ventanas temporales de tamaño 4, separadas 20 minutos entre sí, sobre el área de Nueva York.

El modelo recibe una secuencia de imágenes de radar y trata de predecir la imagen correspondiente al instante siguiente.  
La configuración temporal es la siguiente:

**Entrada al modelo:**  
`t-60min, t-40min, t-20min, t`  

**Predicción:**  
`t+20min`

El entrenamiento se realiza con múltiples ejemplos estructurados de esta manera.

Más detalles técnicos, instrucciones de uso y documentación se añadirán progresivamente con el avance y finalización del proyecto

---

**Proyecto actualmente en desarrollo**  
**Fecha estimada de finalización:** Diciembre de 2025
