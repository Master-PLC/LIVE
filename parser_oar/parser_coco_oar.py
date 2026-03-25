import json
from collections import defaultdict
import sys
import os
from tqdm import tqdm

# 尝试导入用户指定的库
try:
    from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
    # 初始化解析器
    # Factual-Scene-Graph/flan-t5-base-VG-factual-sg
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cuda')
    PARSER_AVAILABLE = True
    print("SceneGraphParser loaded successfully.")
except ImportError:
    print("Warning: 'factual_scene_graph' library not found in the environment.")
    print("Please note that I cannot install new libraries.")
    print("Will proceed with data grouping, but cannot perform scene graph parsing.")
    PARSER_AVAILABLE = False
except Exception as e:
    # 捕捉其他可能的初始化错误 (例如，模型下载失败)
    print(f"Error initializing SceneGraphParser: {e}")
    print("Will proceed with data grouping, but cannot perform scene graph parsing.")
    PARSER_AVAILABLE = False

# 1. 加载输入的 JSON 文件
input_file_path = '/data/home/Yitong/ZJUTruthLab/Hallucination/VLMuncertain/MultiImageBench/COCO/annotations/captions_val2014.json' 
data = None
if not os.path.exists(input_file_path):
    print(f"Error: Input file not found at {input_file_path}")
    print("Please make sure the file 'captions_val2014.json' is available.")
else:
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {input_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# 只有在数据加载成功和解析器可用时才继续
if data and PARSER_AVAILABLE:
    # 2. 预处理：按 image_id 分组所有 captions
    image_id_to_captions = defaultdict(list)
    if 'annotations' in data:
        for ann in data['annotations']:
            image_id_to_captions[ann['image_id']].append(ann['caption'])
        print(f"Grouped captions for {len(image_id_to_captions)} images.")
    else:
        print("Warning: 'annotations' key not found in JSON data.")

    # 3. 预处理：映射 image_id 到 file_name (用户称之为 'base_path')
    image_id_to_filename = {}
    if 'images' in data:
        for img in data['images']:
            image_id_to_filename[img['id']] = img['file_name']
        print(f"Mapped filenames for {len(image_id_to_filename)} images.")
    else:
        print("Warning: 'images' key not found in JSON data.")
        
    # 4. 主处理循环
    all_results = []
    
    # 警告：处理所有图像可能会非常慢。
    # 我们将只处理少量图像（例如前 10 个有标题的图像）作为演示。
    processed_count = 0
    max_images_to_process = 10 
    
    print(f"\nStarting processing. Will process a maximum of {max_images_to_process} images (due to performance limits).")

    for image_id, file_name in tqdm(image_id_to_filename.items()):
        # if processed_count >= max_images_to_process:
        #     print(f"\nReached processing limit of {max_images_to_process} images.")
        #     break
            
        captions = image_id_to_captions.get(image_id)
        
        # 仅处理具有有效标题列表的图像
        if not captions:
            continue
            
        print(f"Processing image_id: {image_id} (File: {file_name}) with {len(captions)} caption(s)...")

        # 用于去重的集合，符合用户要求
        combined_objects = set()
        combined_attributes = defaultdict(set) # 映射 object_name -> set of attributes
        combined_relations = set()
        
        # 解析此图像的所有标题
        try:
            # 一次性将所有标题传递给解析器
            graphs = parser.parse(captions, beam_size=5, return_text=False, max_output_len=512)
            
            # 遍历每个标题返回的图
            for graph_obj in graphs:
                if not graph_obj or 'entities' not in graph_obj:
                    # 解析器可能对某些标题返回空结果
                    continue
                    
                # 临时列表，用于将此图中的索引映射到实体头部名称
                entity_names_in_graph = [entity['head'] for entity in graph_obj['entities']]
                
                # 处理实体（对象和属性）
                for entity in graph_obj.get('entities', []):
                    obj_name = entity.get('head')
                    if not obj_name:
                        continue
                        
                    combined_objects.add(obj_name)
                    
                    attrs = entity.get('attributes', set())
                    if attrs:
                        # 确保 attrs 是一个可迭代对象 (例如，它是一个集合)
                        if isinstance(attrs, set):
                            combined_attributes[obj_name].update(attrs)
                        
                # 处理关系
                for relation in graph_obj.get('relations', []):
                    try:
                        subj_idx = relation['subject']
                        obj_idx = relation['object']
                        rel_verb = relation['relation']
                        
                        # 使用我们的临时映射表查找名称
                        subj_name = entity_names_in_graph[subj_idx]
                        obj_name = entity_names_in_graph[obj_idx]
                        
                        # 格式化为用户请求的字符串
                        relation_str = f"{subj_name} {rel_verb} {obj_name}"
                        combined_relations.add(relation_str)
                    except (IndexError, KeyError, TypeError):
                        # 捕获索引超出范围或键丢失的错误
                        print(f"Warning: Skipping malformed relation for image {image_id}: {relation}")
                        continue

        except Exception as e:
            print(f"Error during parsing captions for image {image_id}: {e}")
            # 继续处理下一张图片
            continue

        # 5. 整理此图像的结果（将集合转换为排序后的列表）
        # (确保属性字典的值也是列表)
        final_attributes = {
            obj: sorted(list(attrs)) 
            for obj, attrs in combined_attributes.items()
        }
        
        image_result = {
            'file_name': file_name, # 用户要求的 'base_path'
            'objects': sorted(list(combined_objects)),
            'attributes': final_attributes,
            'relations': sorted(list(combined_relations)),
            'captions': captions
        }
        
        all_results.append(image_result)
        processed_count += 1

    # 6. 将所有合并的结果保存到一个新的 JSON 文件
    output_file_path = 'coco_parser_oar.json'
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
            
        print(f"\nSuccessfully processed {processed_count} images.")
        print(f"Results saved to {output_file_path}")
        
        # 打印第一个结果作为示例
        if all_results:
            print("\n--- 示例结果 (第一个处理的图像) ---")
            print(json.dumps(all_results[0], indent=4, ensure_ascii=False))
        else:
            print("\nNo images were processed successfully.")

    except Exception as e:
        print(f"Error writing output JSON file: {e}")

elif not data:
     print("Skipping processing because input file 'captions_val2014.json' could not be loaded.")
elif not PARSER_AVAILABLE:
    print("Skipping processing because the 'factual_scene_graph' library is not available.")
    print("Cannot proceed with parsing captions.")