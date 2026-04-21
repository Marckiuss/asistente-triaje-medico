# Usamos una imagen ligera de Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos solo los requisitos primero para aprovechar la caché de Docker
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código de la API
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando para arrancar la API
CMD ["python", "main.py"]