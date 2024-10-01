# Passo 1: Configurando o ambiente
!git clone https://github.com/ultralytics/yolov5  # Clonando o repositório YOLOv5
%cd yolov5
!pip install -r requirements.txt  # Instalando as dependências

# Passo 2: Carregar o modelo YOLOv5 pré-treinado
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Modelo pequeno (s), use 'yolov5m', 'yolov5l', 'yolov5x' para maior precisão, mas mais lento

# Passo 3: Carregar uma imagem para detectar objetos
# Carregue sua imagem do seu computador ou use uma URL
from PIL import Image
import requests
from io import BytesIO

image_url = 'https://ultralytics.com/images/zidane.jpg'  # URL da imagem
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Passo 4: Realizar a detecção de objetos
results = model(img)

# Passo 5: Exibir os resultados
results.print()  # Exibir os resultados no console
results.show()   # Mostrar a imagem com as detecções
