from aesthetic_predictor import predict_aesthetic
from PIL import Image
print(predict_aesthetic(Image.open("bestImages/quality_5/00000-1671415246.jpg")))