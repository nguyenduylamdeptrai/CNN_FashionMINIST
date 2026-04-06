import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Định nghĩa lại đúng kiến trúc model (phải giống hệt trên Kaggle)
class CNN_Fashion(nn.Module):
    def __init__(self):
        super(CNN_Fashion, self).__init__()

        # Khối 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.mp1= nn.MaxPool2d(2)

        # Khối 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.mp2= nn.MaxPool2d(2)

        # Khối 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Fully Connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.relu_fc = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.drop(x)
        x = self.fc2(x)
        logits = x
        return logits



# 2. Load model đã train
device = torch.device('cpu')  # Dùng CPU trên máy local
model = CNN_Fashion().to(device)
model.load_state_dict(torch.load('fashion_cnn.pth', map_location=device))
model.eval()

# 3. Danh sách nhãn
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 4. Hàm dự đoán
def predict(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),       # đảo màu
        transforms.Normalize((0.5,), (0.5,))       # THÊM DÒNG NÀY cho khớp lúc train
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    plt.imshow(img_tensor.squeeze(), cmap='gray')
    plt.title(f'{class_names[pred]} ({conf*100:.1f}%)')
    plt.axis('off')
    plt.show()

# 5. Sử dụng
predict('test_tshirt.jpg')  # Đổi thành đường dẫn ảnh của bạn