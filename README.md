
![img](https://picx.zhimg.com/80/v2-d66d3db596f04aa5e0cae2eaeb0e422c_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

分类名: (图片张数， 标注个数)

ddan_ tt_ vatla:(2416,2523)

ddan_ ct_ vatla:(1955,2014)

总数:(4371, 4537)

![img](https://pic1.zhimg.com/80/v2-3d83112de19379374b938fa1891825f8_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

![img](https://picx.zhimg.com/80/v2-75cc66e0aca05a494a27d71797a30b8d_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

### 电线覆盖物检测数据集统计表

| 标签名称                        | 图片数量 | 边框数量 |
| ------------------------------- | -------- | -------- |
| 电线覆盖物类型1 (ddan_tt_vatla) | 2416     | 2523     |
| 电线覆盖物类型2 (ddan_ct_vatla) | 1955     | 2014     |
| 总计                            | 4371     | 4537     |

### 项目结构

```
深色版本
wire_covering_detection/
├── dataset/
│   ├── images/
│   │   └── *.jpg
│   ├── labels_voc/
│   │   └── *.xml
│   ├── labels_yolo/
│   │   └── *.txt
├── models/
│   └── yolov8/
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   ├── convert_voc_to_yolo.py
├── weights/
│   └── best_model.pt
├── requirements.txt
└── README.md
```

### 1. 安装依赖

首先，确保你已经安装了必要的库。创建一个requirements.txt文件，内容如下：

```
深色版本
torch
torchvision
numpy
pandas
matplotlib
tqdm
pyyaml
opencv-python
ultralytics
lxml
```

然后，使用以下命令安装依赖：

```
bash深色版本
pip install -r requirements.txt
```

### 2. 数据集准备

确保你的数据集已经按照以下结构组织：

```
深色版本
dataset/
├── images/
│   └── *.jpg
├── labels_voc/
│   └── *.xml
├── labels_yolo/
│   └── *.txt
```

每个文件夹中包含对应的图像文件和标签文件。确保所有图像文件都是.jpg格式，标签文件是VOC格式的.xml文件和YOLO格式的.txt文件。

### 3. 数据集转换工具

为了方便使用YOLOv8模型，我们需要将VOC格式的标注文件转换为YOLO格式。可以使用以下脚本进行转换。

### 3.1 src/convert_voc_to_yolo.py

```
python深色版本
import os
import xml.etree.ElementTree as ET
import shutil

def convert_voc_to_yolo(voc_dir, yolo_dir, class_names, image_size=(640, 640)):
    os.makedirs(yolo_dir, exist_ok=True)

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()

        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        yolo_file = os.path.join(yolo_dir, xml_file.replace('.xml', '.txt'))
        with open(yolo_file, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue
                class_id = class_names.index(class_name)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                x_center = (xmin + xmax) / 2.0 / image_width
                y_center = (ymin + ymax) / 2.0 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    voc_dir = "dataset/labels_voc"
    yolo_dir = "dataset/labels_yolo"
    class_names = ["ddan_tt_vatla", "ddan_ct_vatla"]
    convert_voc_to_yolo(voc_dir, yolo_dir, class_names)
```

### 4. 数据集配置

创建一个数据集类，用于加载和预处理数据。

### 4.1 src/utils.py

```
python深色版本
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class WireCoveringDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.label_files = os.listdir(label_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.image_files[index].replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            labels = [line.strip().split() for line in labels]
            labels = [[int(label[0])] + list(map(float, label[1:])) for label in labels]
        else:
            labels = []

        if self.transform:
            image = self.transform(image)

        return image, labels

def get_data_loaders(image_dir, label_dir, batch_size=16, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = WireCoveringDetectionDataset(image_dir, label_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
```

### 5. 模型定义

使用YOLOv8模型进行检测任务。这里我们使用预训练的YOLOv8模型，并对其进行微调。

### 5.1 src/train.py

```
python深色版本
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO
from src.utils import get_data_loaders
import torch.nn.functional as F

def train_model(image_dir, label_dir, epochs=100, batch_size=16, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model
    model = model.to(device)

    train_loader, val_loader = get_data_loaders(image_dir, label_dir, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            labels = [torch.tensor(label, dtype=torch.float32).to(device) for label in labels]

            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.0
            for i in range(len(labels)):
                if len(labels[i]) > 0:
                    pred = outputs[i, :, :, :5 * len(labels[i])]
                    pred = pred.permute(2, 0, 1).reshape(-1, 5)
                    target = labels[i][:, 1:]
                    loss += criterion(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Training Loss', train_loss, epoch)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = [torch.tensor(label, dtype=torch.float32).to(device) for label in labels]

                outputs = model(images)
                loss = 0.0
                for i in range(len(labels)):
                    if len(labels[i]) > 0:
                        pred = outputs[i, :, :, :5 * len(labels[i])]
                        pred = pred.permute(2, 0, 1).reshape(-1, 5)
                        target = labels[i][:, 1:]
                        loss += criterion(pred, target)

                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', val_loss, epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.save("weights/best_model.pt")
    writer.close()

if __name__ == "__main__":
    image_dir = "dataset/images"
    label_dir = "dataset/labels_yolo"
    train_model(image_dir, label_dir)
```

### 6. 模型评估

训练完成后，可以通过验证集和测试集来评估模型的性能。示例如下：

### 6.1 src/predict.py

```
python深色版本
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from src.utils import get_data_loaders
import numpy as np

def predict_and_plot(image_dir, label_dir, model_path, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO(model_path)
    model = model.to(device)
    model.eval()

    _, val_loader = get_data_loaders(image_dir, label_dir)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            labels = [torch.tensor(label, dtype=torch.float32).to(device) for label in labels]

            outputs = model(images)
            outputs = outputs.cpu().numpy()

            images = images.cpu().numpy().transpose((0, 2, 3, 1))
            labels = [label.cpu().numpy() for label in labels]

            for j in range(len(images)):
                ax = axes[j] if num_samples > 1 else axes
                ax[0].imshow(images[j])
                ax[0].set_title(f"True: {labels[j]}")
                ax[0].axis('off')

                ax[1].imshow(images[j])
                ax[1].set_title(f"Predicted: {outputs[j]}")
                ax[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_dir = "dataset/images"
    label_dir = "dataset/labels_yolo"
    model_path = "weights/best_model.pt"
    predict_and_plot(image_dir, label_dir, model_path)
```

### 7. 运行项目

1. 确保你的数据集已经放在相应的文件夹中。
2. 运行转换脚本将VOC格式的标注文件转换为YOLO格式：

```
bash深色版本
python src/convert_voc_to_yolo.py
```

1. 在项目根目录下运行以下命令启动训练：

```
bash深色版本
python src/train.py
```

1. 训练完成后，运行以下命令进行评估和可视化：

```
bash深色版本
python src/predict.py
```

### 8. 功能说明

- **数据集转换**：convert_voc_to_yolo.py脚本用于将VOC格式的标注文件转换为YOLO格式。
- **数据集类**：WireCoveringDetectionDataset类用于加载和预处理数据。
- **数据加载器**：get_data_loaders函数用于创建训练和验证数据加载器。
- **训练模型**：train.py脚本用于训练YOLOv8模型，使用均方误差损失函数和Adam优化器。
- **评估模型**：predict.py脚本用于评估模型性能，并可视化输入图像、真实标签和预测结果。

### 9. 详细注释

### convert_voc_to_yolo.py

- **数据集转换**：将VOC格式的标注文件转换为YOLO格式。

### utils.py

- **数据集类**：定义了一个WireCoveringDetectionDataset类，用于加载和预处理数据。
- **数据加载器**：定义了一个get_data_loaders函数，用于创建训练和验证数据加载器。

### train.py

- **训练函数**：定义了一个train_model函数，用于训练YOLOv8模型。
- **训练过程**：在每个epoch中，模型在训练集上进行前向传播和反向传播，并在验证集上进行评估。

### predict.py

- **预测和可视化**：定义了一个predict_and_plot函数，用于在验证集上进行预测，并可视化输入图像、真实标签和预测结果。