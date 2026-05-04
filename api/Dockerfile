FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalación forzada de dependencias críticas para evitar problemas de compatibilidad
RUN pip install --no-cache-dir tensorflow==2.16.1 keras==3.14.0 keras-nlp==0.28.0 keras-hub==0.28.0

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]