# Imagen base
FROM python:3.12.3

# Creamos una carpeta
RUN mkdir /app

# La establecemos como directorio de trabajo
WORKDIR /api

# Añadimos el contenido de app a app
ADD ./api /api
ADD ./requirements.txt /api

# Instalamos las dependencias de python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY / ./

# Ejecutamos el comando para lanzar la API
CMD ["fastapi", "run", "./api/api.py","--port=8000"]

# Exponemos el puerto del backend
EXPOSE 8000