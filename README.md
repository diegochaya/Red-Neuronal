# Red-Neuronal

El proyecto comenzó como un perceptrón muy sencillo, y quise avanzar a una Red Neuronal capaz de encontrar patrones en problemas más complejos. Tras 2 días de pelearme con subínidces, salió una primera red un poco primitiva, pero de tamaño no predefinido, es decir, uno de los inputs era una lista donde cada elemento i representaba el número de neuronas en la capa i. Esa red fue mejorando, se añadió los bias, funciones de activación en cada capa y varias mejoras más. La actual está escrita de manera matricial lo que dificulta su comprensión, pero resulta ser mucho más rápido para numpy. 
La Red Neuronal ha sido probada con reconocimiento de números en imágenes y otros ejemplos menos comunes como detección de arritmias en señales o predicción de números primos (este último no obtuvo muy buenos resultados). 
En el código actual solo adjunto la Red Neuronal en sí. ¡Espero que os guste! 
