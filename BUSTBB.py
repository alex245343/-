import cv2
import numpy as np
import sqlite3
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import gradio as gr
import os
import threading

Image.MAX_IMAGE_PIXELS = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def load_image_from_path(path):
    if not os.path.isfile(path):
        print(f"Файл не нашёл: {path}")
        return None
    
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Ошибка с чтением изображения: {path}. Error: {e}")
        return None

def resize_image(img, target_size=(256, 256)):
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize(target_size, Image.LANCZOS)
    return np.array(img_resized)

def compare_images(img1, img2):
    target_size = (256, 256)
    img1_resized = resize_image(img1, target_size)
    img2_resized = resize_image(img2, target_size)
    
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def load_requirements_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            запрещённые_элементы = set(line.strip() for line in file)
        return запрещённые_элементы
    except Exception as e:
        print(f"не грузит требования: {e}")
        return set()

def check_description(description, запрещённые_элементы):
    lower_description = description.lower().strip()
    
    lower_description = ''.join(c for c in lower_description if c.isalnum() or c.isspace())

    if any(word in lower_description for word in запрещённые_элементы):
        return "Не соответствует требованиям ВОЗ: присутствуют запрещённые элементы"
    
    return "Соответствует требованиям ВОЗ"

def process_image(item, user_img, base_path, запрещённые_элементы, results, lock):
    img_id, img_name, img_path, description = item
    full_path = os.path.join(base_path, img_path)
    
    if not os.path.isfile(full_path):
        print(f"Файл не нашёлся: {full_path}")
        return
    
    db_img = load_image_from_path(full_path)
    
    if db_img is None:
        return

    try:
        score = compare_images(user_img, db_img)
    except Exception as e:
        print(f"Не грузит фото?: {e}")
        return

    print(f"АЙди: {img_id}, Имя: {img_name}")
    print(f"Скор: {score:.2f}")
    print("-" * 30)

    description_check = check_description(description, запрещённые_элементы)
    with lock:
        if score >= 0.8:
            results['match'] = (
                f"Match found!\nID: {img_id}\nName: {img_name}\nDescription: {description}\nPath: {full_path}\nScore: {score:.2f}",
                description_check
            )
            results['stop'] = True
        elif score > results['highest_score']:
            results['highest_score'] = score
            results['best_match'] = (img_id, img_name, description, full_path, score)
            results['description_info'] = description_check

def find_best_match(user_image):
    conn = sqlite3.connect(os.path.join(SCRIPT_DIR, 'products.db'))
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, image_path, description FROM products")
    database = cursor.fetchall()
    conn.close()
    
    user_img = np.array(user_image)
    base_path = SCRIPT_DIR

    запрещённые_элементы = load_requirements_from_txt(os.path.join(SCRIPT_DIR, 'TR.txt'))

    results = {
        'match': None,
        'best_match': None,
        'highest_score': 0,
        'description_info': "",
        'stop': False
    }
    lock = threading.Lock()
    
    threads = []
    for item in database:
        if results['stop']:
            break
        thread = threading.Thread(target=process_image, args=(item, user_img, base_path, запрещённые_элементы, results, lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if results['match']:
        return results['match']
    elif results['best_match']:
        img_id, img_name, description, img_path, score = results['best_match']
        return (
            f"No match above 80%, but the closest match found:\nID: {img_id}\nName: {img_name}\nDescription: {description}\nPath: {img_path}\nScore: {score:.2f}",
            results['description_info']
        )
    else:
        return (
            "No match found.",
            results['description_info']
        )

iface = gr.Interface(
    fn=find_best_match,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Подробная информация"), gr.Textbox(label="Оценка ВОЗ")],
    title="Image Matcher",
    description="Загрузите изображалку"
)

iface.launch()

print("Запрещённые элементы:")
запрещённые_элементы = load_requirements_from_txt(os.path.join(SCRIPT_DIR, 'TR.txt'))
for элемент in запрещённые_элементы:
    print(элемент)
