import os
import shutil
import random
from tqdm import tqdm  # 导入tqdm

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def build_subset(source_dir, target_dir, test_dir, images_per_id, id_limit, test_samples):
    # 确保输出文件夹存在
    clear_directory(target_dir)
    clear_directory(test_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "negative"), exist_ok=True)

    selected_set = set()

    # 获取所有身份目录
    identity_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # 1. 首先创建训练子集（face_subset）
    if len(identity_dirs) < id_limit:
        selected_dirs = identity_dirs
    else:
        selected_dirs = random.sample(identity_dirs, id_limit)

    # 使用tqdm包装循环
    for identity in tqdm(selected_dirs, desc="创建训练子集"):
        src_path = os.path.join(source_dir, identity)
        images = os.listdir(src_path)
        if len(images) >= images_per_id:
            dst_path = os.path.join(target_dir, identity)
            os.makedirs(dst_path, exist_ok=True)
            selected_images = random.sample(images, images_per_id)
            for img in selected_images:
                shutil.copy(os.path.join(src_path, img), os.path.join(dst_path, img))
                selected_set.add(f"{identity}_{img}")
    print(f"\n训练子集创建完成: {id_limit}个人物, 每人{images_per_id}张图像")

    # 2. 创建测试集
    # 正样本（来自selected_dirs中的人物，但不同图片）
    positive_count = 0
    with tqdm(total=test_samples, desc="生成正样本") as pbar:
        while positive_count < test_samples:
            identity = random.choice(selected_dirs)
            src_path = os.path.join(source_dir, identity)
            images = os.listdir(src_path)
            img = random.choice(images)
            while f"{identity}_{img}" in selected_set:
                img = random.choice(images)
            selected_set.add(f"{identity}_{img}")
            shutil.copy(os.path.join(src_path, img), os.path.join(test_dir, "positive", f"{identity}_{img}"))
            positive_count += 1
            pbar.update(1)

    # 负样本（来自不在selected_dirs中的人物）
    negative_dirs = [d for d in identity_dirs if d not in selected_dirs]
    random.shuffle(negative_dirs)
    negative_count = 0
    negative_selected_set = set()
    with tqdm(total=test_samples, desc="生成负样本") as pbar:
        while negative_count < test_samples:
            identity = random.choice(negative_dirs)
            src_path = os.path.join(source_dir, identity)
            images = os.listdir(src_path)
            img = random.choice(images)
            while f"{identity}_{img}" in negative_selected_set:
                img = random.choice(images)
            selected_set.add(f"{identity}_{img}")
            shutil.copy(os.path.join(src_path, img), os.path.join(test_dir, "negative", f"{identity}_{img}"))
            negative_count += 1
            pbar.update(1)

    print(f"\n测试集创建完成: {positive_count}正样本, {negative_count}负样本")


if __name__ == "__main__":
    # 原始数据路径
    SOURCE_DIR = "D:/data/archive/VGG-Face2/data/vggface2_train/train"
    # 目标子集路径
    TARGET_DIR = "face_subset"
    # 测试集路径
    TEST_DIR = "./face_testset"
    # 每个人保留的最大图像数
    IMAGES_PER_ID = 10
    # 保留的身份数量
    ID_LIMIT = 1000
    # 测试集样本数量
    TEST_SAMPLES = 1000

    build_subset(SOURCE_DIR, TARGET_DIR, TEST_DIR, IMAGES_PER_ID, ID_LIMIT, TEST_SAMPLES)