from transformers import ViTImageProcessor, ViTModel
from PIL import Image

image = Image.open("path/to/your/image.jpg")

processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
