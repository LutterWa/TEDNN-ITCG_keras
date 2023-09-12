# 基于迁移-集成的时间约束计算制导算法
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)

1. 运行data_generator.py, 采集预训练模型和新任务的数据
2. 打开新任务数据，确定前两条弹道的数据长度
3. 修改tednn.py中的tl0和tl1为第1条弹道和第2条弹道长度，然后运行脚本，训练预训练模型和迁移模型
4. 最后运行itcg.py进行测试