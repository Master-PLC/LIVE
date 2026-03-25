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

# !! 核心参数 !!
IMAGE_DIR = "/data/home/Yitong/ZJUTruthLab/Hallucination/VLMuncertain/del/MultiImageBench/COCO/val2014"
BATCH_SIZE = 256         # 提取特征时的批次大小

# --- K值实验的配置 ---
# 1. 在这里定义您想测试的K值
# COCO val 一共有 40,504 images
K_VALUES_TO_TEST = [12, 80, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
save_base_dir = "./k_experiments_results"
K_EXPERIMENT_RESULTS_FILE = os.path.join(save_base_dir, 'k_experiment_results.json')
K_PLOT_FILE = os.path.join(save_base_dir,'k_analysis_plot.png')
K_PLOT_PDF = os.path.join(save_base_dir, 'k_analysis_plot.pdf')

# --- 输出文件路径 ---
# (特征文件是共享的)
EMBEDDINGS_FILE = os.path.join(save_base_dir,'data/data_clip_embeddings_hf.npy')
IMAGE_PATHS_FILE = os.path.join(save_base_dir,'data/data_image_paths_hf.json')

# --- 设备设置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 使用设备: {DEVICE} ---")

# --- 2. Phase 1: 特征提取 (已修改为 Transformers) ---

def extract_all_features():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IMAGE_PATHS_FILE):
        print(f"'{EMBEDDINGS_FILE}' 和 '{IMAGE_PATHS_FILE}' 已存在，跳过特征提取。")
        return
    print("--- Phase 1: 开始提取CLIP特征 (使用 Transformers) ---")
    print(f"加载模型: openai/clip-vit-base-patch32 ...")
    # 按照您的要求加载模型
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    model.to(DEVICE).eval() # 移动到设备并设置为评估模式
    # 查找所有图片
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
        # 预加载和预处理图片
        for img_path in batch_paths:
            try:
                # 使用 .convert('RGB') 来处理灰度图等异常
                img = Image.open(img_path).convert('RGB')
                valid_images.append(img)
                valid_paths_in_batch.append(img_path)
            except Exception as e:
                print(f"警告: 跳过无法加载的图片 {img_path} - {e}")
        if not valid_images:
            continue
        # 使用 processor 批量处理图片
        inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(DEVICE)
        # 提取特征
        with torch.no_grad():
            # AutoModelForZeroShotImageClassification 的结构是:
            # .vision_model -> 原始视觉输出
            # .visual_projection -> 投影层
            # 我们需要投影后的特征，这才是标准的CLIP特征
            vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
            pooled_output = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled_output)
            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
        all_features.append(image_features.cpu().numpy())
        all_valid_paths.extend(valid_paths_in_batch)
    # 合并为一个大的 Numpy 数组
    final_features = np.vstack(all_features)
    print(f"成功提取了 {len(all_valid_paths)} 张图片的特征。")
    print(f"特征矩阵形状: {final_features.shape}")
    # 保存到磁盘
    np.save(EMBEDDINGS_FILE, final_features)
    with open(IMAGE_PATHS_FILE, 'w') as f:
        json.dump(all_valid_paths, f)
    print(f"特征已保存到 '{EMBEDDINGS_FILE}'")
    print(f"图片路径已保存到 '{IMAGE_PATHS_FILE}'")

# --- 3. K值观察实验 ---
def find_optimal_k():
    if os.path.exists(K_EXPERIMENT_RESULTS_FILE):
        print(f"'{K_EXPERIMENT_RESULTS_FILE}' 已存在，跳过K值实验。")
        print("--- K值实验历史结果 ---")
        with open(K_EXPERIMENT_RESULTS_FILE, 'r') as f:
            results = json.load(f)
        print_k_results_table(results)
        # 即使文件存在，也尝试绘图
        plot_k_results(results)
        return
    print("--- Phase 1.5: 开始 K 值观察实验 ---")
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"错误: '{EMBEDDINGS_FILE}' 未找到。请先运行特征提取。")
        return
    print("加载特征...")
    embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
    n_samples, d = embeddings.shape
    print(f"加载了 {n_samples} 个 {d}-维 特征。")
    gpu_res = None
    use_gpu = False
    if DEVICE == "cuda":
        try:
            # 检查 'StandardGpuResources' 属性是否存在
            if hasattr(faiss, 'StandardGpuResources'):
                gpu_res = faiss.StandardGpuResources()
                use_gpu = True
                print("FAISS: 已成功分配GPU资源 (faiss-gpu)。")
            else:
                print("FAISS: 检测到 faiss-cpu 版本，退回到CPU。")
        except Exception as e:
            print(f"FAISS: 无法分配GPU资源 ({e})，退回到CPU。")
    results = []
    # 确保 K 值不超过样本数
    valid_k_values = [k for k in K_VALUES_TO_TEST if k <= n_samples]
    if len(valid_k_values) < len(K_VALUES_TO_TEST):
        print(f"警告: K值 { [k for k in K_VALUES_TO_TEST if k > n_samples] } 大于样本数，已跳过。")
    for k in valid_k_values:
        print(f"\n--- 正在测试 K = {k} ---")
        start_time = time.time()
        kmeans = faiss.Kmeans(d=d, k=k, niter=200, nredo=5, verbose=True, gpu=use_gpu)
        print(f"  (FAISS 警告是正常的，正在训练 K={k}...)")
        kmeans.train(embeddings)
        
        # 2. 分配并统计簇大小
        # D 是到质心的 平方 L2 距离
        D, I = kmeans.index.search(embeddings, 1)
        cluster_assignments = I.flatten()
        # WCSS 是所有点到其各自质心的平方距离之和
        inertia = float(D.sum()) # 转换为
        size_counts = defaultdict(int)
        unique_ids, counts = np.unique(cluster_assignments, return_counts=True)
        for size in counts:
            size_counts[size] += 1
            
        # 3. 统计可用簇
        num_ge_50 = sum(v for k_size, v in size_counts.items() if k_size >= 50) # <--- 新增
        num_ge_4 = sum(v for k_size, v in size_counts.items() if k_size >= 4)
        num_ge_3 = sum(v for k_size, v in size_counts.items() if k_size >= 3)
        num_ge_2 = sum(v for k_size, v in size_counts.items() if k_size >= 2)
        num_eq_1 = size_counts.get(1, 0)
        
        end_time = time.time()
        
        result_entry = {
            "K": k,
            "Inertia (WCSS)": inertia,
            "Clusters_ge_50": num_ge_50, # <--- 新增
            "Clusters_ge_4": num_ge_4,
            "Clusters_ge_3": num_ge_3,
            "Clusters_ge_2": num_ge_2,
            "Clusters_eq_1": num_eq_1,
            "Time_sec": end_time - start_time
        }
        results.append(result_entry)
        print(f"K={k} 完成，耗时: {end_time - start_time:.2f} 秒。Inertia: {inertia:.2f}, 簇>=50: {num_ge_50}, 簇>=4: {num_ge_4}")

    # 释放GPU
    if gpu_res:
        gpu_res.noDelete = False
        print("FAISS: 已释放GPU资源。")

    # 打印最终表格
    print("\n--- K值实验最终结果 ---")
    print_k_results_table(results)

    # 保存结果
    with open(K_EXPERIMENT_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n实验结果已保存到 '{K_EXPERIMENT_RESULTS_FILE}'")
    
    # 自动绘图
    plot_k_results(results)


def print_k_results_table(results):
    """辅助函数：打印漂亮的表格"""
    # 打印表头
    print(f"{'K':>6} | {'Inertia (WCSS)':>18} | {'簇 >= 50':>8} | {'簇 >= 4':>8} | {'簇 >= 3':>8} | {'簇 >= 2':>8} | {'簇 = 1':>8}")
    print("-" * 84) # <--- 修改
    # 打印数据
    for entry in results:
        # 使用 .get('Clusters_ge_50', 0) 确保在读取旧文件时不会崩溃
        print(f"{entry['K']:>6} | {entry['Inertia (WCSS)']:>18.2f} | {entry.get('Clusters_ge_50', 0):>8} | {entry['Clusters_ge_4']:>8} | {entry['Clusters_ge_3']:>8} | {entry['Clusters_ge_2']:>8} | {entry['Clusters_eq_1']:>8}")


def plot_k_results(results):
    """
    (新功能) 自动绘制 K 值实验结果并保存到文件。
    """
    print(f"\n--- 正在绘制结果曲线并保存到 '{K_PLOT_FILE}' ---")
    try:
        plt.rcParams['font.size'] = 14           # 基础字体大小（原默认约为10）
        plt.rcParams['axes.labelsize'] = 16      # 坐标轴标签 (X/Y Label) 大小
        plt.rcParams['axes.titlesize'] = 18      # 标题 (Title) 大小
        plt.rcParams['xtick.labelsize'] = 14     # X轴刻度数字大小
        plt.rcParams['ytick.labelsize'] = 14     # Y轴刻度数字大小
        plt.rcParams['legend.fontsize'] = 14     # 图例字体大小
        plt.rcParams['font.weight'] = 'bold'     # 全局字体加粗
        plt.rcParams['axes.labelweight'] = 'bold' # 坐标轴标签加粗
        plt.rcParams['axes.titleweight'] = 'bold' # 标题加粗

        k_values = [r['K'] for r in results]
        inertia_values = [r['Inertia (WCSS)'] for r in results]
        clusters_ge_4 = [r['Clusters_ge_4'] for r in results]
        # 使用 .get 确保旧的 results.json 文件也能被（部分）绘制
        clusters_ge_50 = [r.get('Clusters_ge_50', 0) for r in results] # <--- 新增
        clusters_eq_1 = [r['Clusters_eq_1'] for r in results]

        # 创建一个包含两个子图的画布
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- 绘制图1: Inertia (肘部法则) ---
        color = 'tab:blue'
        # --- 修改：字体加粗 ---
        # ax1.set_xlabel('Number of Clusters', fontweight='bold')
        ax1.set_ylabel('WCSS', color=color, fontweight='bold')
        
        ax1.set_xlabel('Number of Clusters K', fontweight='bold')
        

        ax1.plot(k_values, inertia_values, marker='o', color=color, label='WCSS')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title('K-means', fontweight='bold') # <--- 修改：字体加粗

        # --- 绘制图2: 可用簇数量 (共享Y轴) ---
        ax2 = ax1.twinx() # 共享 X 轴
        color = 'tab:green'
        # --- 修改：字体加粗 ---
        # ax2.set_ylabel('Number of Clusters', color=color, fontweight='bold')
        ax2.set_ylabel('Number of Valid Clusters', color=color, fontweight='bold')

        # --- 修改：绘制 >= 4 和 >= 50 ---
        ax2.plot(k_values, clusters_ge_4, marker='s', color=color, linestyle='--', label='cluster size >= 4')
        ax2.plot(k_values, clusters_ge_50, marker='D', color=color, linestyle=':', label='Cluster Size >= 50') 
        
        # ax2.plot(k_values, clusters_eq_1, marker='x', color='tab:red', linestyle=':', label='cluster size = 1 (single sample)')
        ax2.tick_params(axis='y', labelcolor=color)

        # --- 修改：图例字体加粗 ---
        fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85), prop={'weight': 'bold'})
        fig.tight_layout() # 调整布局防止标签重叠
        
        # 保存图像
        plt.savefig(K_PLOT_FILE)
        plt.savefig(K_PLOT_PDF, format='pdf', bbox_inches='tight')
        print(f"绘图成功，已保存到 '{K_PLOT_FILE}'")

    except Exception as e:
        print(f"\n警告：绘图失败。您可能需要安装 matplotlib (pip install matplotlib)。错误: {e}")

if __name__ == "__main__":
    # 步骤 1: 提取特征 (如果文件不存在)
    extract_all_features()
    # 步骤 2: 运行 K 值观察实验 (如果文件不存在)
    find_optimal_k()