import os
import glob
import json
import random
import time
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import faiss # 用于高速聚类
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 1. 配置 ---
IMAGE_DIR = "/data/home/Yitong/ZJUTruthLab/Hallucination/VLMuncertain/MultiImageBench/COCO/val2014"
BATCH_SIZE = 256           # 提取特征时的批次大小

CHOSEN_K = 1000           
SAMPLES_PER_TYPE = 500

# --- 新增：采样策略配置 ---
# 簇间采样：从距离最近的 10% 簇中随机采样
CLOSE_CLUSTER_RATIO = 0.1 
# 簇内采样：从距离（质心）最远的 10% 图像中随机采样
FAR_IMAGE_RATIO = 0.1     
# 候选池的最小数量，防止比例太小导致池子为空
MIN_CLOSE_CLUSTER_POOL = 10 
MIN_FAR_IMAGE_POOL = 5 

# (特征文件是共享的)
k_experiments_base_dir = "./k_experiments_results"
EMBEDDINGS_FILE = os.path.join(k_experiments_base_dir,'data/data_clip_embeddings_hf.npy')
IMAGE_PATHS_FILE = os.path.join(k_experiments_base_dir,'data/data_image_paths_hf.json')

save_base_dir = "./split_images"
def get_cluster_output_files(k):
    """根据K值生成动态文件名"""
    base_k = f"_k{k}"
    return {
        "cluster_assign": os.path.join(save_base_dir,f'data_cluster_assignments{base_k}.npy'),
        "inverted_index": os.path.join(save_base_dir,f'cluster_inverted_index{base_k}.json'),
        "final_samples": os.path.join(save_base_dir,f'sampled_all_tuples{base_k}.json'),
        "cluster_centroids": os.path.join(save_base_dir,f'data_cluster_centroids{base_k}.npy')
    }

# --- 设备设置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 使用设备: {DEVICE} ---")

def extract_all_features():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IMAGE_PATHS_FILE):
        print(f"'{EMBEDDINGS_FILE}' 和 '{IMAGE_PATHS_FILE}' 已存在，跳过特征提取。")
        return
    print("--- Phase 1: 开始提取CLIP特征 (使用 Transformers) ---")
    print(f"加载模型: openai/clip-vit-base-patch32 ...")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    model.to(DEVICE).eval()
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"在 '{IMAGE_DIR}' 中未找到 *.jpg 文件。请检查路径。")
    print(f"在 '{IMAGE_DIR}' 中找到 {len(image_paths)} 张图片。")
    all_features = []
    all_valid_paths = []
    print(f"开始批量提取特征 (Batch size: {BATCH_SIZE})...")
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="特征提取"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        valid_images = []
        valid_paths_in_batch = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                valid_images.append(img)
                valid_paths_in_batch.append(img_path)
            except Exception as e:
                print(f"警告: 跳过无法加载的图片 {img_path} - {e}")
        if not valid_images:
            continue
        inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
            pooled_output = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled_output)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        all_features.append(image_features.cpu().numpy())
        all_valid_paths.extend(valid_paths_in_batch)
    final_features = np.vstack(all_features)
    print(f"成功提取了 {len(all_valid_paths)} 张图片的特征。")
    print(f"特征矩阵形状: {final_features.shape}")
    np.save(EMBEDDINGS_FILE, final_features)
    with open(IMAGE_PATHS_FILE, 'w') as f:
        json.dump(all_valid_paths, f)
    print(f"特征已保存到 '{EMBEDDINGS_FILE}'")
    print(f"图片路径已保存到 '{IMAGE_PATHS_FILE}'")


# --- 4. Phase 2: 聚类并建立索引 (使用选定的K) ---
def run_clustering_and_indexing(k):
    """
    加载特征，使用 *选定的K* 运行 FAISS K-Means，建立倒排索引，并保存质心。
    """
    output_files = get_cluster_output_files(k)
    INVERTED_INDEX_FILE = output_files["inverted_index"]
    CLUSTER_ASSIGN_FILE = output_files["cluster_assign"]
    # --- 新增 ---
    CLUSTER_CENTROIDS_FILE = output_files["cluster_centroids"]

    # --- 修改：同时检查质心文件 ---
    if os.path.exists(INVERTED_INDEX_FILE) and os.path.exists(CLUSTER_CENTROIDS_FILE):
        print(f"'{INVERTED_INDEX_FILE}' 和 '{CLUSTER_CENTROIDS_FILE}' (K={k}) 已存在，跳过聚类和索引。")
        return

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"错误: '{EMBEDDINGS_FILE}' 未找到。请先运行特征提取。")
        return
        
    print(f"--- Phase 2: 开始聚类并建立索引 (K={k}) ---")
    print("加载特征...")
    embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
    with open(IMAGE_PATHS_FILE, 'r') as f:
        image_paths = json.load(f)

    if embeddings.shape[0] != len(image_paths):
        raise ValueError("特征数量和路径数量不匹配！")

    n_samples, d = embeddings.shape
    print(f"加载了 {n_samples} 个 {d}-维 特征。")

    print(f"开始 FAISS K-Means 聚类 (K={k})...")
    gpu_res = None
    if DEVICE == "cuda":
        try:
            gpu_res = faiss.StandardGpuResources()
            print("FAISS: 已分配GPU资源。")
        except Exception as e:
            print(f"FAISS: 无法分配GPU资源 ({e})，退回到CPU。")
    
    kmeans = faiss.Kmeans(d=d, k=k, niter=200, nredo=5, verbose=True, gpu=(gpu_res is not None))
    kmeans.train(embeddings)
    print("K-Means 训练完成。")

    # --- 新增：保存质心 ---
    centroids = kmeans.centroids
    np.save(CLUSTER_CENTROIDS_FILE, centroids)
    print(f"质心已保存到 '{CLUSTER_CENTROIDS_FILE}'")

    print("为所有图片分配簇ID...")
    D, I = kmeans.index.search(embeddings, 1)
    cluster_assignments = I.flatten()
    
    if gpu_res:
        gpu_res.noDelete = False
        print("FAISS: 已释放GPU资源。")
        
    np.save(CLUSTER_ASSIGN_FILE, cluster_assignments)
    print(f"簇分配结果已保存到 '{CLUSTER_ASSIGN_FILE}'")

    print("建立簇 -> 图片 的倒排索引...")
    cluster_to_images_index = defaultdict(list)
    for img_path, cluster_id in zip(image_paths, cluster_assignments):
        cluster_to_images_index[str(int(cluster_id))].append(img_path)

    with open(INVERTED_INDEX_FILE, 'w') as f:
        json.dump(cluster_to_images_index, f, indent=2)
    print(f"倒排索引已保存到 '{INVERTED_INDEX_FILE}'")
    
    size_counts = defaultdict(int)
    for imgs in cluster_to_images_index.values():
        size_counts[len(imgs)] += 1
    
    print("\n--- 最终聚类统计信息 ---")
    print(f"总簇数: {len(cluster_to_images_index)}")
    print(f"   - 簇大小 >= 4 的簇有: {sum(v for k_size, v in size_counts.items() if k_size >= 4)} 个")
    print(f"   - 簇大小 >= 3 的簇有: {sum(v for k_size, v in size_counts.items() if k_size >= 3)} 个")
    print(f"   - 簇大小 >= 2 的簇有: {sum(v for k_size, v in size_counts.items() if k_size >= 2)} 个")
    print(f"   - 簇大小 = 1 的簇有: {size_counts.get(1, 0)} 个 (孤立点)")
    print("----------------------\n")


# --- 5. Phase 3: 采样 (使用选定的K) ---
def sample_all_tuples(k):
    output_files = get_cluster_output_files(k)
    INVERTED_INDEX_FILE = output_files["inverted_index"]
    FINAL_SAMPLES_FILE = output_files["final_samples"]
    CLUSTER_CENTROIDS_FILE = output_files["cluster_centroids"] # 需加载

    # --- 检查所有必需文件 ---
    required_files = [INVERTED_INDEX_FILE, CLUSTER_CENTROIDS_FILE, EMBEDDINGS_FILE, IMAGE_PATHS_FILE]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 必需文件 '{f}' 未找到。请先运行前面的步骤。")
            return

    if os.path.exists(FINAL_SAMPLES_FILE):
        print(f"'{FINAL_SAMPLES_FILE}' (K={k}) 已存在，跳过采样。")
        return

    print(f"--- Phase 3: 开始策略采样 (K={k}, 每个类型 {SAMPLES_PER_TYPE} 个) ---")

    # --- (A) 加载所有需要的数据 ---
    print("加载索引、质心、特征和路径...")
    with open(INVERTED_INDEX_FILE, 'r') as f:
        cluster_to_images = json.load(f) # { "cluster_id_str": [path1, path2, ...] }
    
    centroids = np.load(CLUSTER_CENTROIDS_FILE).astype('float32')
    embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
    
    with open(IMAGE_PATHS_FILE, 'r') as f:
        all_image_paths = json.load(f)

    print("构建辅助数据结构...")
    # 路径 -> 特征索引
    image_path_to_index = {path: i for i, path in enumerate(all_image_paths)}
    
    # 簇ID (str) -> 簇内图片在全局 embeddings 中的索引 (list of int)
    cluster_id_to_image_indices = defaultdict(list)
    for cluster_id_str, paths in cluster_to_images.items():
        indices = [image_path_to_index[p] for p in paths]
        cluster_id_to_image_indices[cluster_id_str] = indices

    # 簇ID (str) -> 质心数组中的索引 (int)
    if k != len(centroids):
         raise ValueError(f"K值 ({k}) 与加载的质心数量 ({len(centroids)}) 不匹配！")
    
    all_cluster_ids = sorted(cluster_to_images.keys(), key=int)
    cluster_id_to_centroid_index = {cid: int(cid) for cid in all_cluster_ids}
    
    # --- (B) 预计算：簇间距离矩阵 ---
    print("计算 (K, K) 簇间距离矩阵...")
    dist_sq_matrix = faiss.pairwise_distances(centroids, centroids)
    cluster_dist_matrix = np.sqrt(np.clip(dist_sq_matrix, 0, None))
    print(f"簇间距离矩阵计算完成，形状: {cluster_dist_matrix.shape}")

    # --- (C) 定义新的采样辅助函数 ---
    # (这两个辅助函数 `sample_close_clusters` 和 `sample_dispersed_images` 保持不变)

    def sample_close_clusters(base_cluster_id, n_clusters, min_size=1):
        """
        从 base_cluster_id（str）的“近邻”中采样 n_clusters 个簇。
        确保采样的簇满足 min_size。
        """
        base_centroid_idx = cluster_id_to_centroid_index[base_cluster_id]
        dists = cluster_dist_matrix[base_centroid_idx]
        sorted_centroid_indices = np.argsort(dists)
        pool_size = max(MIN_CLOSE_CLUSTER_POOL, int(len(dists) * CLOSE_CLUSTER_RATIO))
        candidate_pool_ids = [] # 存放满足条件的 cluster_id (str)
        
        for centroid_idx in sorted_centroid_indices:
            if centroid_idx == base_centroid_idx:
                continue
            cluster_id_str = all_cluster_ids[centroid_idx]
            if len(cluster_to_images[cluster_id_str]) >= min_size:
                candidate_pool_ids.append(cluster_id_str)
            if len(candidate_pool_ids) >= pool_size:
                break

        if len(candidate_pool_ids) < n_clusters:
            # print(f"警告: 簇 {base_cluster_id} 的 {pool_size} 个近邻中满足 size>={min_size} 的不足 {n_clusters} 个。扩大到全局查找。")
            global_pool = [cid for cid in all_cluster_ids if cid != base_cluster_id and len(cluster_to_images[cid]) >= min_size]
            if len(global_pool) < n_clusters:
                raise ValueError(f"全局也找不到 {n_clusters} 个 size>={min_size} 的簇 (排除 {base_cluster_id})")
            return random.sample(global_pool, n_clusters)
        
        return random.sample(candidate_pool_ids, n_clusters)

    def sample_dispersed_images(cluster_id, n_images):
        """
        从 cluster_id (str) 中采样 n_images 个“分散”的图片。
        """
        image_paths = cluster_to_images[cluster_id]
        image_indices = cluster_id_to_image_indices[cluster_id]
        cluster_features = embeddings[image_indices]
        
        if len(image_paths) < n_images:
            raise ValueError(f"簇 {cluster_id} 只有 {len(image_paths)} 张图片，无法采样 {n_images} 张。")

        sampled_paths = []
        sampled_features = []
        
        first_idx_in_cluster = random.choice(range(len(image_paths)))
        sampled_paths.append(image_paths[first_idx_in_cluster])
        sampled_features.append(cluster_features[first_idx_in_cluster])
        
        for _ in range(n_images - 1):
            if len(sampled_features) == 1:
                anchor_vec = sampled_features[0:1] # (1, d)
            else:
                anchor_vec = np.mean(np.array(sampled_features), axis=0, keepdims=True)
            
            dist_sq_to_anchor = faiss.pairwise_distances(anchor_vec, cluster_features).flatten()
            pool_size = max(MIN_FAR_IMAGE_POOL, int(len(image_paths) * FAR_IMAGE_RATIO))
            sorted_indices_in_cluster = np.argsort(dist_sq_to_anchor)
            
            candidate_pool_paths = []
            for idx_in_cluster in reversed(sorted_indices_in_cluster):
                path = image_paths[idx_in_cluster]
                if path not in sampled_paths:
                    candidate_pool_paths.append(path)
                if len(candidate_pool_paths) >= pool_size:
                    break
            
            if not candidate_pool_paths:
                remaining_paths = [p for p in image_paths if p not in sampled_paths]
                if not remaining_paths:
                    break
                chosen_path = random.choice(remaining_paths)
            else:
                chosen_path = random.choice(candidate_pool_paths)

            sampled_paths.append(chosen_path)
            chosen_idx_in_cluster = image_paths.index(chosen_path)
            sampled_features.append(cluster_features[chosen_idx_in_cluster])

        # 确保返回了正确数量的图片，如果中途 break 导致数量不够，则抛出异常
        if len(sampled_paths) != n_images:
             raise ValueError(f"簇 {cluster_id} 内部采样 {n_images} 个分散图片失败，只得到 {len(sampled_paths)} 个。")
             
        return sampled_paths

    # --- (D) 重新执行采样循环 (增加唯一性检查) ---

    clusters_ge_4 = [cid for cid, imgs in cluster_to_images.items() if len(imgs) >= 4]
    clusters_ge_3 = [cid for cid, imgs in cluster_to_images.items() if len(imgs) >= 3]
    clusters_ge_2 = [cid for cid, imgs in cluster_to_images.items() if len(imgs) >= 2]
    all_clusters = list(cluster_to_images.keys())
    
    print(f"可用簇 (size>=4): {len(clusters_ge_4)}, (size>=3): {len(clusters_ge_3)}, (size>=2): {len(clusters_ge_2)}, (all): {len(all_clusters)}")
    
    if len(all_clusters) < 4:
        raise ValueError(f"簇太少 ({len(all_clusters)})，无法采样 ABCD。")
    # (省略其他检查...)

    all_samples = []
    # --- 新增：全局去重集合 ---
    seen_image_sets = set()
    
    # SAMPLE_TYPES = [
    #     'AA', 'AB', 'AAA', 'AAB', 'ABC',
    #     'AAAA', 'AAAB', 'AABB', 'AABC', 'ABCD'
    # ]
    SAMPLE_TYPES = [
        'AAAA', 'ABCD'
    ]

    print(f"为每种模式生成 {SAMPLES_PER_TYPE} 个 *独特* 样本...")
    
    for sample_type in SAMPLE_TYPES:
        print(f"   正在生成 {sample_type} ...")
        samples_for_type = []
        
        # --- 新增：重试循环逻辑 ---
        # 每个目标样本最多尝试 100 次，总尝试次数为 SAMPLES_PER_TYPE * 100
        MAX_TOTAL_ATTEMPTS = SAMPLES_PER_TYPE * 100 
        attempts = 0

        pbar = tqdm(total=SAMPLES_PER_TYPE, desc=f"   {sample_type}")

        while len(samples_for_type) < SAMPLES_PER_TYPE and attempts < MAX_TOTAL_ATTEMPTS:
            attempts += 1
            try:
                images = []
                cluster_ids = []
                
                # --- 2-image ---
                if sample_type == 'AA':
                    c_A = random.choice(clusters_ge_2)
                    images = sample_dispersed_images(c_A, 2)
                    cluster_ids = [c_A] * 2
                
                elif sample_type == 'AB':
                    c_A = random.choice(all_clusters)
                    c_B = sample_close_clusters(c_A, 1, min_size=1)[0]
                    images = [random.choice(cluster_to_images[c_A]), 
                              random.choice(cluster_to_images[c_B])]
                    cluster_ids = [c_A, c_B]

                # --- 3-image ---
                elif sample_type == 'AAA':
                    c_A = random.choice(clusters_ge_3)
                    images = sample_dispersed_images(c_A, 3)
                    cluster_ids = [c_A] * 3

                elif sample_type == 'AAB':
                    c_A = random.choice(clusters_ge_2)
                    c_B = sample_close_clusters(c_A, 1, min_size=1)[0]
                    images = sample_dispersed_images(c_A, 2)
                    images.append(random.choice(cluster_to_images[c_B]))
                    cluster_ids = [c_A, c_A, c_B]

                elif sample_type == 'ABC':
                    c_A = random.choice(all_clusters)
                    c_B, c_C = sample_close_clusters(c_A, 2, min_size=1)
                    images = [random.choice(cluster_to_images[c_A]),
                              random.choice(cluster_to_images[c_B]),
                              random.choice(cluster_to_images[c_C])]
                    cluster_ids = [c_A, c_B, c_C]

                # --- 4-image ---
                elif sample_type == 'AAAA':
                    c_A = random.choice(clusters_ge_4)
                    images = sample_dispersed_images(c_A, 4)
                    cluster_ids = [c_A] * 4

                elif sample_type == 'AAAB':
                    c_A = random.choice(clusters_ge_3)
                    c_B = sample_close_clusters(c_A, 1, min_size=1)[0]
                    images = sample_dispersed_images(c_A, 3)
                    images.append(random.choice(cluster_to_images[c_B]))
                    cluster_ids = [c_A, c_A, c_A, c_B]

                elif sample_type == 'AABB':
                    c_A = random.choice(clusters_ge_2)
                    c_B = sample_close_clusters(c_A, 1, min_size=2)[0] 
                    images = sample_dispersed_images(c_A, 2)
                    images.extend(sample_dispersed_images(c_B, 2))
                    cluster_ids = [c_A, c_A, c_B, c_B]

                elif sample_type == 'AABC':
                    c_A = random.choice(clusters_ge_2)
                    c_B, c_C = sample_close_clusters(c_A, 2, min_size=1)
                    images = sample_dispersed_images(c_A, 2)
                    images.append(random.choice(cluster_to_images[c_B]))
                    images.append(random.choice(cluster_to_images[c_C]))
                    cluster_ids = [c_A, c_A, c_B, c_C]

                elif sample_type == 'ABCD':
                    c_A = random.choice(all_clusters)
                    c_B, c_C, c_D = sample_close_clusters(c_A, 3, min_size=1)
                    images = [random.choice(cluster_to_images[c_A]),
                              random.choice(cluster_to_images[c_B]),
                              random.choice(cluster_to_images[c_C]),
                              random.choice(cluster_to_images[c_D])]
                    cluster_ids = [c_A, c_B, c_C, c_D]
                
                # --- 新增：唯一性检查 ---
                if not images: # 如果采样失败，images为空
                    continue
                    
                image_set = frozenset(images)

                # 确保它是一个有效的、包含多张图片的集合
                if len(image_set) < 2: 
                    continue

                if image_set not in seen_image_sets:
                    # 这是一个全新的组合！
                    seen_image_sets.add(image_set)
                    samples_for_type.append({
                        "type": sample_type,
                        "image_paths": images, 
                        "cluster_ids": cluster_ids
                    })
                    pbar.update(1) # 进度条前进
                # else:
                    # 如果 image_set 已经在 seen_image_sets 中，
                    # 我们什么都不做，循环将继续下一次尝试

            except (ValueError, IndexError) as e:
                # 捕获 (ValueError: 找不到足够的簇) 或 (IndexError: 采样[0]失败)
                # 这只是一次失败的尝试，而不是一个致命错误
                # print(f"警告: 采样 {sample_type} 时出错: {e}") # (打印太多，注释掉)
                pass # 仅计为一次失败的尝试

        pbar.close()
        # --- (结束 while 循环) ---

        all_samples.extend(samples_for_type)
        
        # 循环结束，检查是否达到了目标
        if len(samples_for_type) < SAMPLES_PER_TYPE:
            print(f" 警告: 无法为 {sample_type} 生成 {SAMPLES_PER_TYPE} 个独特样本。")
            print(f" (在 {MAX_TOTAL_ATTEMPTS} 次尝试后只找到了 {len(samples_for_type)} 个)")
        else:
            print(f" 成功生成 {len(samples_for_type)} 个 {sample_type} 样本 ")

    # --- (E) 保存最终结果 ---
    print(f"\n总共生成 {len(all_samples)} 个独特样本。")
    print(f"保存到 '{FINAL_SAMPLES_FILE}' ...")
    
    with open(FINAL_SAMPLES_FILE, 'w') as f:
        json.dump(all_samples, f, indent=2)
        
    print("--- Phase 3: 策略采样完成 ---")


if __name__ == "__main__":
    # extract_all_features()
    print(f"\n--- 使用选定的 K = {CHOSEN_K} 执行后续步骤 ---")
    run_clustering_and_indexing(CHOSEN_K)
    sample_all_tuples(CHOSEN_K)
    print(f"\n--- K={CHOSEN_K} 的所有流程完成 ---")