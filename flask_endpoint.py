import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms, models
import json

# Завантаження моделі
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Завантаження посилань із JSON-файлу
with open('wiki_reference.json', 'r') as json_file:
    class_links = json.load(json_file)


# Класи
classes = ['agaric', 'agaricus', 'boletus', 'earthstar', 'gyromitra', 'russula', 'stinkhorn']

# Трансформації
mean = [0.4143, 0.3704, 0.2914]
std = [0.2253, 0.2066, 0.1967]

image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Класифікація зображення
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream).convert("RGB")
        image = image_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        class_name = classes[predicted_class.item()]
        link = class_links.get(class_name, "No link available")  # Отримання посилання для класу

        print(classes[predicted_class.item()], round(confidence.item() * 100, 2))
        return jsonify({
            "class_name": classes[predicted_class.item()],
            "confidence": round(confidence.item() * 100, 2),
            "link": link
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Запуск Flask-сервера
if __name__ == '__main__':
    app.run(debug=True)
