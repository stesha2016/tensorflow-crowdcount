# crowd count(人流密度算法)
 * [参考算法](https://github.com/svishwa/crowdcount-mcnn) 主要是参考此算法中提到的论文用tensorflow进行实现。
 * 可以下载参考算法中提到的数据集，用matlab进行处理后，用本算法进行training
## 如何训练自己的数据集
 * 参考算法中没有提到这部分的处理，本文会做一个介绍
 1. 准备自己的数据集图片
 2. 用[LabelImg](https://github.com/tzutalin/labelImg)对数据集图片进行标注，标注方法就是将人头框出来label写入head即可。
 3. 用[get_density_map](https://github.com/stesha2016/tensorflow-crowdcount/blob/master/data_preparation/get_density_map.py)文件进行解析，会生成scv文件
 4. 修改training文件中的文件路径就可以开始训练了。
## 性能
 * 对于参考算法中提供的patches_A_9数据集，训练1000个epoch，MAE会下降到9.7。效果并不算太好，因为对于场景不固定的情况下准确率并不算太高。
 * 训练自己的数据集，在固定站台拍摄，场景单一，learning_rate为0.00005，200个epoch后，MAE可以到1.9，差不多可以达到90%以上的准确率，所以如果是用在固定场景效果会更好。
## 结果
 * 预测可以得到图片的人头数，还可以得到density的图片。根据density的图片可以得到标注人头的图片
 ![分屏标记](https://github.com/stesha2016/tensorflow-crowdcount/blob/master/images/test.jpg)
