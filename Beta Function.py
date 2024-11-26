import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

# Путь к папке с изображениями
image_dir = 'C:/Users/Huawei/Desktop/фотографии с дрона/'

# Получение списка всех изображений в папке
all_files = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, fname) for fname in all_files if fname.lower().endswith('.jpg')]
print(f"Найдено {len(image_paths)} изображений.")
print(f"Список изображений: {image_paths}")

# Преобразование изображений для обработки
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Изменение размера изображения до 256x256
    transforms.ToTensor()  # Преобразование изображения в тензор
])

# Загрузка и преобразование изображений в тензоры
images = []
for img_path in image_paths:
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image)
    images.append(image_tensor)

# Преобразование тензоров в numpy массивы
images_np = [img.permute(1, 2, 0).numpy() * 255 for img in images]  # Перемещение каналов и масштабирование

# Создание масок полей
field_masks = []
for img in images_np:
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Применение адаптивного порога
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Применение морфологических операций для улучшения маски
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Дополнительная фильтрация по размеру контура
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Фильтрация по площади контура
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    field_masks.append(mask)

# Создание RGBA изображений с прозрачными областями полей
rgba_images = []
for i, img in enumerate(images_np):
    mask = field_masks[i]
    rgba_image = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba_image[..., :3] = img
    rgba_image[..., 3] = mask  # Прозрачные области становятся непрозрачными
    rgba_images.append(rgba_image)

# Визуализация результатов обработки для первых двух изображений
plt.figure(figsize=(12, 6))

for i in range(min(2, len(image_paths))):
    # Оригинальное изображение
    plt.subplot(2, 2, 2 * i + 1)
    plt.imshow(images_np[i].astype(np.uint8))
    plt.title(f"Original Image {i + 1}")
    plt.axis('off')

    # RGBA изображение с прозрачными полями
    plt.subplot(2, 2, 2 * i + 2)
    plt.imshow(rgba_images[i])
    plt.title(f"Transparent Fields Image {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Путь к папке для сохранения изображений
output_dir = 'C:/Users/Huawei/Desktop/processed_images/'
os.makedirs(output_dir, exist_ok=True)

# Сохранение обработанных изображений
for i, rgba_img in enumerate(rgba_images):
    rgba_img_pil = Image.fromarray(rgba_img)
    rgba_img_pil.save(os.path.join(output_dir, f'detected_fields_{i + 1}.png'))

print(f"Обработанные изображения сохранены в папке: {output_dir}")
