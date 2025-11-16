# Imagen base de Python
FROM python:3.10-slim

# Evitar que Python genere .pyc y habilitar logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema mínimas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente (lo que esté en la raíz del proyecto)
COPY . .

# Exponer puerto de FastAPI
EXPOSE 8000

# Comando por defecto: levantar FastAPI con Uvicorn
# El entrypoint será app/main.py con objeto app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
