# Guion del video (máx. 5 min, ~650 palabras habladas)

Formato: pitch con la app de Streamlit y 2-3 gráficos en pantalla. No mostrar código. Los valores `[XX]` se rellenan con `output/tables/` cuando termine la corrida final; no inventes ni redondees a favor.

---

## 1. El problema (0:00–0:40)
**Pantalla:** Tab 1 de la app, o una diapositiva con una imagen de CV.

"Un equipo de reclutamiento recibe miles de CVs. Saber cuántos años de experiencia tiene cada candidato ayuda a precargar su perfil y a ordenar el trabajo. El problema es que casi nunca dice '12 años de experiencia': hay que deducirlo de las fechas de cada empleo, de cuántos cargos hubo y de cuándo terminó la carrera. Un reclutador lo hace de cabeza. Mi proyecto enseña a un sistema a hacerlo desde la imagen del CV, y mide cuánto mejor lo hace un modelo avanzado frente a métodos simples."

## 2. De imagen a texto (0:40–1:20)
**Pantalla:** Tab 2, imagen del CV con las cajas de PaddleOCR al lado del texto.

"El CV llega como imagen, así que primero lo leo con PaddleOCR. Cada línea sale con su texto, su confianza y su posición, y con las posiciones reconstruyo el orden de lectura, incluso con dos columnas, para que un cargo no se separe de sus fechas. Usé [XX] CVs sintéticos del dataset PopResume, [XX] por ocupación, solo sin foto. Frente al texto original, el OCR tiene un error de caracteres de [XX] y lee bien el [XX]% de los años. Eso importa porque un dígito mal leído cambia la respuesta. También borré nombre, contacto y cualquier frase que diga la experiencia o la edad directamente: se enmascararon en [XX] CVs."

## 3. Por qué es difícil (1:20–2:00)
**Pantalla:** un CV real del test, resaltando fechas y cargos.

"Miren este CV. En ningún lado dice cuántos años tiene. Hay que sumar 2019 a 2021, 2021 a 2023, descontar solapes y huecos, y mirar cuándo se graduó. El modelo tiene que conectar información repartida por todo el documento, no basta con buscar palabras clave."

## 4. Cuatro formas de adivinar (2:00–2:50)
**Pantalla:** diapositiva con cuatro íconos.

"Comparé cuatro modelos con el mismo reparto de datos. Uno: la mediana, que siempre dice lo mismo; es el piso. Dos: TF-IDF, que cuenta qué palabras aparecen, sin entender el orden. Tres: embeddings, que resumen el CV en un vector con un modelo ya entrenado y congelado. Cuatro: BERT afinado, que lee todo el texto y aprende a relacionar fechas y cargos. Para los CVs largos conservo el inicio y el final, porque ahí están los empleos más recientes y la educación."

## 5. Resultados (2:50–3:50)
**Pantalla:** Tab 3: tabla de comparación y gráfico predicho vs real.

"En el conjunto de prueba, el error medio fue de [XX] años para la mediana, [XX] para TF-IDF, [XX] para embeddings y [XX] para BERT. BERT acierta dentro de ±2 años en el [XX]% de los casos. Su mejora frente a TF-IDF es de [XX] años, con un intervalo de confianza de [XX] a [XX]: [sí / no] es una mejora estadísticamente clara. En el gráfico, los puntos cercanos a la diagonal son aciertos; [describir si hay tendencia a acercarse al centro]. Un buen caso: [CV, real vs predicho]. Un mal caso: [CV, real vs predicho], causado por [fecha mal leída / empleos solapados / hueco / truncado]."

## 6. ¿Vale la pena? ¿Es justo? (3:50–4:45)
**Pantalla:** tabla de equidad y diapositiva final.

"¿Compensa BERT? Cuesta [XX] veces más de entrenar y [XX] de inferencia que TF-IDF, a cambio de [XX] años menos de error. [Conclusión honesta con los números.] En la revisión de equidad, el error medio de BERT por sexo y por raza fue [XX]; [hay / no hay] un sesgo sistemático, aunque esos atributos no son entradas del modelo. Limitaciones: CVs sintéticos, [XX] ocupaciones, estadísticas de EE. UU. y solo inglés. Este sistema sirve para precargar un perfil o ayudar a ordenar trabajo, pero nunca debe decidir quién es contratado."

## 7. Cierre (4:45–5:00)
"En resumen: la imagen se convierte en texto con PaddleOCR y BERT [lee mejor / no mejora mucho] las fechas que los métodos simples. Gracias."

---

### Consejos
- Ensaya con cronómetro; si te pasas, recorta la sección 3.
- Muestra la app corriendo (`python -m streamlit run app.py`) en lugar de capturas.
- Di cada número con su contexto ("error medio de X años"), no solo la cifra.
- Cuando tengas el video, pega el enlace en `video/link.txt`.
