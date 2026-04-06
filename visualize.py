import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# === 1. Định nghĩa model (copy từ file train) ===
class CNN_Fashion(nn.Module):
    def __init__(self):
        super(CNN_Fashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

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
        return x

# === 2. Load model ===
device = torch.device('cpu')
model = CNN_Fashion().to(device)
model.load_state_dict(torch.load('fashion_cnn.pth', map_location=device))
model.eval()

# === 3. Tiền xử lý ảnh ===
img = Image.open('test_tshirt.jpg').convert('L')
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.5,), (0.5,))
])
img_tensor = transform(img).unsqueeze(0).to(device)

# === 4. Trích xuất feature maps ===
with torch.no_grad():
    feature_maps = model.relu1(model.bn1(model.conv1(img_tensor)))
    x = model.mp1(feature_maps)  # qua MaxPool trước
    feature_maps2 = model.relu2(model.bn2(model.conv2(x)))

# === 5. Vẽ 32 feature maps ===
# fig, axes = plt.subplots(4, 8, figsize=(16, 8))
# fig.suptitle('32 Feature Maps từ lớp Conv2d đầu tiên', fontsize=16)



# for i, ax in enumerate(axes.flat):
#     ax.imshow(feature_maps[0, i].cpu(), cmap='gray')
#     ax.set_title(f'Filter {i}', fontsize=8)
#     ax.axis('off')

# Vẽ 64 feature maps
fig, axes = plt.subplots(8, 8, figsize=(16, 16))
fig.suptitle('64 Feature Maps từ lớp Conv2d thứ 2', fontsize=16)    

for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps2[0, i].cpu(), cmap='gray')
    ax.set_title(f'Filter {i}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()