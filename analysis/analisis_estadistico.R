# 1. CARGAR LIBRERÍAS
library(tidyverse)
library(ggplot2)
library(tidytext)

# 2. CONFIGURAR RUTA (Basada en tu getwd())
csv_path <- "data/processed/Symptom2Disease_final.csv"

# 3. LEER DATOS
if (!file.exists(csv_path)) {
  stop(paste("ERROR: No se encuentra el archivo en:", csv_path))
}

df <- read.csv(csv_path)

# 4. PROCESAR
df$numero_palabras <- sapply(strsplit(as.character(df$text), " "), length)

# --- GRÁFICO 1 ---
p1 <- ggplot(df, aes(x = fct_infreq(label))) +
        geom_bar(fill = "steelblue", color = "black") +
        coord_flip() +
        labs(title = "Casos registrados por Especialidad", x = "Enfermedad", y = "Casos") +
        theme_minimal()

# --- GRÁFICO 2 ---
p2 <- ggplot(df, aes(x = reorder(label, numero_palabras, FUN = median), y = numero_palabras)) +
        geom_boxplot(fill = "lightgreen", color = "darkgreen") +
        coord_flip() +
        labs(title = "Longitud de síntomas", x = "Enfermedad", y = "Palabras") +
        theme_minimal()

# --- GRÁFICO 3 ---
df_words <- df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

p3 <- df_words %>%
  count(word, sort = TRUE) %>%
  top_n(15) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "coral", color = "black") +
  coord_flip() +
  labs(title = "Top 15 palabras repetidas", x = "Palabra", y = "Frecuencia") +
  theme_minimal()

# 5. GUARDAR (En la raíz donde estás ahora)
ggsave("analysis/plots/01_distribucion_enfermedades.png", plot = p1, width = 10, height = 6)
ggsave("analysis/plots/02_longitud_sintomas.png", plot = p2, width = 10, height = 8)
ggsave("analysis/plots/03_palabras_frecuentes.png", plot = p3, width = 10, height = 6)

print("GRÁFICOS GENERADOS CON ÉXITO")
