# IJCAI21-WhoIsWho
IJCAI21-WhoIsWho 比赛代码



# Usage

数据存放于`data/v3/`文件夹(v3表示whoiswho第三版数据，v1和v2数据可在 https://www.aminer.cn/whoiswho 下载)
其中：
`train`存放训练数据
`cna_data`存放验证集数据
`cna_test_data`存放测试集数据
`processed`存放经过处理后的文件


1. 特征工程，运行
```
1-Feature_engineering.ipynb
```
生成训练集、验证机、测试集特征

2. 训练分类器
```
2-Train_classifiers.ipynb
```
训练不同的分类器用于集成

3. 推理
```
3-Inference.ipynb
```
模型集成推理，输出结果

# NOTE
`get_sementic_train_data.py`, `get_sementic_valid_data.py`, `get_sementic_test_data.py`在这里的代码没有用到，主要是用于提取数据的语义特征，增加后可以稍微提升模型效果，如果需要进一步提升模型效果可以自行加入。