import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from torchvision import transforms
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext

# Функции для преобразования из RGB в LAB
def rgb2xyz(rgb):
    mask = (rgb > 0.04045).astype(float)
    rgb = (((rgb + 0.055) / 1.055) ** 2.4) * mask + (rgb / 12.92) * (1 - mask)
    rgb = rgb * 100
    xyz = np.zeros_like(rgb)
    xyz[..., 0] = rgb[..., 0] * 0.4124564 + rgb[..., 1] * 0.3575761 + rgb[..., 2] * 0.1804375
    xyz[..., 1] = rgb[..., 0] * 0.2126729 + rgb[..., 1] * 0.7151522 + rgb[..., 2] * 0.0721750
    xyz[..., 2] = rgb[..., 0] * 0.0193339 + rgb[..., 1] * 0.1191920 + rgb[..., 2] * 0.9503041
    return xyz

def xyz2lab(xyz):
    ref_white = np.array([95.047, 100.000, 108.883])
    xyz = xyz / ref_white
    mask = (xyz > 0.008856).astype(float)
    xyz = (xyz ** (1 / 3)) * mask + (7.787 * xyz + 16 / 116) * (1 - mask)
    lab = np.zeros_like(xyz)
    lab[..., 0] = (116 * xyz[..., 1]) - 16
    lab[..., 1] = 500 * (xyz[..., 0] - xyz[..., 1])
    lab[..., 2] = 200 * (xyz[..., 1] - xyz[..., 2])
    return lab

def rgb2lab(rgb):
    xyz = rgb2xyz(rgb)
    lab = xyz2lab(xyz)
    return lab

def calculate_green_percentage(image):
    image_np = np.array(image)
    green_channel = image_np[..., 1]
    green_pixels = np.sum((green_channel > image_np[..., 0]) & (green_channel > image_np[..., 2]))
    total_pixels = image_np.shape[0] * image_np.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100
    return green_percentage

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data:
        gps_info = exif_data.get(34853)
        return gps_info
    return None

def convert_to_degrees(value):
    d = float(value[0].numerator) / float(value[0].denominator)
    m = float(value[1].numerator) / float(value[1].denominator)
    s = float(value[2].numerator) / float(value[2].denominator)
    return d + (m / 60.0) + (s / 3600.0)

def get_coordinates(gps_info):
    if not gps_info:
        return None, None
    gps_latitude = gps_info.get(2)
    gps_latitude_ref = gps_info.get(1)
    gps_longitude = gps_info.get(4)
    gps_longitude_ref = gps_info.get(3)
    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = convert_to_degrees(gps_latitude)
        if gps_latitude_ref != "N":
            lat = -lat
        lon = convert_to_degrees(gps_longitude)
        if gps_longitude_ref != "E":
            lon = -lon
        return lat, lon
    return None, None

async def process_image(image_path, update: Update) -> None:
    image = Image.open(image_path).convert('RGB')

    # Преобразование изображения для обработки
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_np = image_tensor.numpy()

    # Преобразование изображения из RGB в LAB
    img_rgb = image_np.transpose(1, 2, 0)
    img_lab = rgb2lab(img_rgb)

    # Разворачивание изображения для кластеризации
    reshaped_img_lab = img_lab.reshape(-1, 3)

    # Кластеризация с использованием K-means для LAB
    num_clusters = 3
    kmeans_lab = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_lab.fit(reshaped_img_lab)
    cluster_labels_lab = kmeans_lab.labels_


    # Подсчет процента зеленых пикселей
    green_percent = calculate_green_percentage(image_np.transpose(1, 2, 0))
    if green_percent < 2:
        gps_info = get_exif_data(image_path)
        if gps_info:
            latitude, longitude = get_coordinates(gps_info)
            if latitude and longitude:
                await update.message.reply_text(f'Координаты GPS некачественного поля - Широта: {latitude}, Долгота: {longitude}')
            else:
                await update.message.reply_text('Координаты GPS не найдены.')
        else:
            await update.message.reply_text('Метаданные GPS не найдены.')
    else:
        await update.message.reply_text('Изображение прошло проверку на качество.')

# Telegram bot setup
TOKEN = '7430797950:AAHRF10w1mUdpOIajFZCjGBP_8iq7rkovE4'

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Отправьте мне фотографии в виде файлов, и я проверю их на качество.')

async def handle_document(update: Update, context: CallbackContext) -> None:
    document = update.message.document
    file = await context.bot.get_file(document.file_id)
    file_path = os.path.join('downloads', f'{document.file_id}.jpg')
    await file.download_to_drive(file_path)
    await process_image(file_path, update)

def main() -> None:
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.MimeType("image/jpeg"), handle_document))
    application.run_polling()

if __name__ == '__main__':
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    main()
