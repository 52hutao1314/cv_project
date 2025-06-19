import os
import numpy as np
import pickle
from deepface import DeepFace
from tqdm import tqdm
import csv
import random

# ---------- 参数配置 ----------
TOP_K = 5  # 输出前K个最相似的文件夹

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def distance_to_confidence(distance, std_dist, min_d, max_d, lambda_=1.0):
    # 计算置信度，使用线性归一化法计算置信度，引入标准差作为惩罚因子
    penalty = np.exp(-lambda_ * std_dist)
    norm = (distance - min_d) / max(max_d - min_d, 1e-6)
    return round((1 - norm) * penalty, 4)

def build_representations(db_path="face_subset", model_name="VGG-Face", rep_path="representations.pkl"):
    # 构建并保存特征库
    representations = [] # 存放所有向量

    # 获取总人数和总图片数用于进度条
    print("正在扫描人脸数据库...")
    person_list = [p for p in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, p))]
    total_persons = len(person_list)
    total_images = sum([len(os.listdir(os.path.join(db_path, p))) for p in person_list])
    print(f"发现 {total_persons} 个人，共 {total_images} 张图片")
    with tqdm(person_list, desc="处理人物", unit="person") as pbar:
        for person in pbar:
            person_path = os.path.join(db_path, person)
            pbar.set_postfix({"current": person})

            img_list = os.listdir(person_path)
            for img_name in img_list:
                img_path = os.path.join(person_path, img_name)
                try:
                    # 每张图生成特征向量
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name=model_name,
                        enforce_detection=False,
                        detector_backend="retinaface",
                        align=True
                    )[0]["embedding"]

                    representations.append({
                        "identity": os.path.join(person, img_name),
                        "embedding": embedding
                    })
                except Exception as e:
                    print(f"\n[跳过] 处理失败: {img_path}，错误：{e}")

    # 保存为特征向量
    print("正在保存特征向量...")
    with open(rep_path, "wb") as f:
        pickle.dump(representations, f)

def load_representations(rep_path="representations.pkl"):
    # 加载特征库
    with open(rep_path, "rb") as f:
        representations = pickle.load(f)
        return representations

def get_folder_embeddings(rep_path="representations.pkl"):
    """
    从保存的特征表示文件中提取按文件夹分组的特征向量

    参数:
        rep_path (str): 预训练特征表示文件的路径，默认为"representations.pkl"

    返回:
        dict: 以文件夹名为键，该文件夹下所有特征向量为值的字典
             格式: {folder_name: [embedding1, embedding2, ...]}
    """
    folder_embeddings = {}
    representations = load_representations(rep_path)
    for rep in representations:
        identity = rep["identity"]
        folder = os.path.basename(os.path.dirname(identity))  # 获取上级文件夹名
        folder_embeddings.setdefault(folder, []).append(rep["embedding"])
    return folder_embeddings

def recognize(folder_embeddings, query_img="0001_01.jpg", model_name="VGG-Face"):
    """
    人脸识别函数，通过比较查询图像与特征库中的特征，找出最相似的文件夹

    参数:
        folder_embeddings: dict - 以文件夹名为键，该文件夹下所有特征向量为值的字典，由 get_folder_embeddings()方法获取
        query_img: str - 查询图像路径，默认为"0001_01.jpg"
        model_name: str - 使用的模型名称，默认为"VGG-Face"

    返回:
        tuple - (confidence_ratio, distance, 最相似文件夹名)
            confidence_ratio 衡量 第一名和第二名的差距是否显著（值越大，识别越可靠）。
            distance 是查询图片与最相似文件夹的平均距离。
    """
    # ---------- 查询图像特征 ----------
    query_embedding = DeepFace.represent(
        img_path=query_img,
        model_name=model_name,
        enforce_detection=False,
        detector_backend="retinaface",
        align=True
    )[0]["embedding"]

    # ---------- 计算每个文件夹的平均距离和标准差 ----------
    folder_stats = []
    all_distances = []
    for folder, embeddings in folder_embeddings.items():
        distances = [euclidean_distance(query_embedding, emb) for emb in embeddings]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        all_distances.append(mean_dist)
        folder_stats.append({
            "folder": folder,
            "mean_distance": round(mean_dist, 4),
            "std_distance": round(std_dist, 4),
        })

    min_d, max_d = min(all_distances), max(all_distances)

    for folder_stat in folder_stats:
        confidence = distance_to_confidence(folder_stat["mean_distance"], folder_stat["std_distance"], min_d, max_d)
        folder_stat["confidence"] = confidence

    # ---------- 排序 & 输出 ----------
    folder_stats.sort(key=lambda x: -x["confidence"])
    top_folders = folder_stats[:TOP_K]

    # print(f"\nTop {TOP_K} 最相似的文件夹：")
    # for i, f in enumerate(top_folders, 1):
    #     print(f"{i}. 文件夹: {f['folder']}")
    #     print(f"   平均距离: {f['mean_distance']}, 标准差: {f['std_distance']}, 置信度: {f['confidence']}")

    # ---------- 判断识别结果是否可信 ----------
    confidence_difference1 = top_folders[0]["confidence"] - top_folders[1]["confidence"]
    confidence_difference2 = top_folders[1]["confidence"] - top_folders[2]["confidence"]

    confidence_ratio = confidence_difference1 / confidence_difference2
    distance = top_folders[0]["mean_distance"]
    return confidence_ratio, distance, top_folders[0]["folder"]

def batch_recognize(query_folder="face_testset/positive", rep_path="representations.pkl",
                    model_name="VGG-Face", result_path="results.csv", recognize_num=100):
    """
    批量识别文件夹下所有图像，并保存真实标签、预测标签、置信度比值

    参数:
        query_folder: str - 待识别图像所在的文件夹路径
        rep_path: str - 特征库文件路径
        model_name: str - DeepFace 使用的模型
        result_path: str - 保存识别结果的CSV文件路径
    """
    results = []

    folder_embeddings = get_folder_embeddings(rep_path=rep_path)
    # 遍历文件夹中的图像
    img_list = [f for f in os.listdir(query_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    selected_list = random.sample(img_list, recognize_num)

    print(f"\n开始批量识别，共 {len(selected_list)} 张图像")
    for img_name in tqdm(selected_list, desc="识别图像", unit="image"):
        img_path = os.path.join(query_folder, img_name)

        # 提取真实标签
        if "_" in img_name:
            true_label = img_name.split("_")[0]
        else:
            true_label = "unknown"

        try:
            # 识别图像
            confidence_ratio, distance, predicted_label = recognize(
                folder_embeddings=folder_embeddings,
                query_img=img_path,
                model_name=model_name
            )
        except Exception as e:
            print(f"[跳过] {img_name} 识别失败: {e}")
            predicted_label = "error"
            confidence_ratio = -1
            distance = -1

        results.append({
            "image": img_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence_ratio": round(confidence_ratio, 4),
            "distance": round(distance, 4),
        })

    # 保存为CSV文件
    with open(result_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_label", "predicted_label", "confidence_ratio", "distance"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n识别完成，结果已保存至：{result_path}")


if __name__ == "__main__":
    # build_representations(db_path="face_subset", model_name="VGG-Face", rep_path="representations.pkl")

    # folder_embeddings = get_folder_embeddings(rep_path="representations.pkl")
    # success, file_name = recognize(folder_embeddings, query_img="face_testset/negative/n006237_0200_01.jpg", model_name="VGG-Face")
    # print(success, file_name)

    batch_recognize(query_folder="face_testset/positive", rep_path="representations.pkl",
                    model_name="VGG-Face", result_path="positive_results.csv", recognize_num=1000)

    batch_recognize(query_folder="face_testset/negative", rep_path="representations.pkl",
                    model_name="VGG-Face", result_path="negative_results.csv", recognize_num=1000)