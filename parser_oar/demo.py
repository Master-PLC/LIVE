from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
# Default parser for single sentences or simple text
parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
# text_graph = parser.parse(["2 beautiful pigs are flying on the sky with 2 bags on their backs"], 
#                          beam_size=1, return_text=True)
graph_obj = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs"], 
                        beam_size=5, return_text=False, max_output_len=128)
# print(text_graph[0])
# Output: ( pigs , is , 2 ) , ( pigs , is , beautiful ) , ( bags , on back of , pigs ) , ( pigs , fly on , sky ) , ( bags , is , 2 )
from factual_scene_graph.utils import tprint
print(graph_obj)
# tprint(graph_obj[0])



from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')

graph_obj = parser.parse(["A bicycle replica with a clock as the front wheel."], 
                        beam_size=5, return_text=True, max_output_len=128)
print(graph_obj) # [{'entities': [{'head': 'bicycle', 'quantity': '', 'attributes': {'replica'}}, {'head': 'wheel', 'quantity': '', 'attributes': {'front', 'clock'}}], 'relations': [{'subject': 0, 'relation': 'have', 'object': 1}]}]
# 需要解析出 object list、 attribute list 以及 relation list
# 比如对于这个例子，得到的结果就是
# object: ['bicycle', 'wheel']
# attribute: {'bicycle': ['replica'], 'wheel': ['front', 'clock']}
# relation: ['bicycle have wheel']



graph_obj = parser.parse(["A bicycle replica with a clock as the front wheel."], 
                        beam_size=5, return_text=False, max_output_len=128)
tprint(graph_obj[0])

# Entities:
# +----------+------------+--------------+
# | Entity   | Quantity   | Attributes   |
# |----------+------------+--------------|
# | bicycle  |            | replica      |
# | wheel    |            | clock,front  |
# +----------+------------+--------------+
# Relations:
# +-----------+------------+----------+
# | Subject   | Relation   | Object   |
# |-----------+------------+----------|
# | bicycle   | have       | wheel    |
# +-----------+------------+----------+

# 1234
# 1、2、3、4
# 12、13、14、23、 in image 1 and imge 2

# 混淆矩阵画一下
# 1、2、image 去对数量做消融
# 3、4 image去变换图片顺序做消融

# 多图的重要性。
# 内生

# 
# 123、134、234、124
# 1234


# Entities:
# +----------+------------+--------------+
# | Entity   | Quantity   | Attributes   |
# |----------+------------+--------------|
# | bicycle  |            | replica      |
# | wheel    |            | clock,front  |
# +----------+------------+--------------+
# Relations:
# +-----------+------------+----------+
# | Subject   | Relation   | Object   |
# |-----------+------------+----------|
# | bicycle   | have       | wheel    |
# +-----------+------------+----------+

