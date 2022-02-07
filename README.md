## Text Normalization（文本正则化）

任务描述：通过机器学习算法将英文文本的“手写”形式转换成“口语“形式，例如“6ft”转换成“six feet”等

### 目前实验成果

1. XGBoost + bag-of-words: 0.99159
2. XGBoost+Weights+rules：0.99002
3. 进阶solve函数（使用6个output文件）：0.98939
4. 基本solve函数：0.98277
5. RandomTree + Rules：0.95304
6. XGboost：0.92605

参考github网址：

- https://www.kaggle.com/c/text-normalization-challenge-english-language （Task Description, Dataset）
- https://github.com/applenob/text_normalization
- https://github.com/songzy12/Text-Normalization
- https://www.kaggle.com/alphasis/xgboost-with-context-label-data-acc-99-637

数据来源网址：

- https://www.kaggle.com/c/text-normalization-challenge-english-language
- https://www.kaggle.com/google-nlu/text-normalization 

数据分析EDA网址（帮助快速理解数据特征）：

- https://www.kaggle.com/allunia/eda-en-text-normalization
- https://www.kaggle.com/headsortails/watch-your-language-update-feature-engineering

### 提升点

#### 1. 数据的不平衡性

对于平衡的数据，我们一般使用准确率作为一般的评估标准（accuracy），当类别不平衡时，准确率就具有迷惑性，而且意义不大。因此有以下几种主流评测标准

- Receiver operating curve，计算ROC曲线面积（二分类，从PLAIN和非PLAIN）
- Precision-recall curve，计算此曲线下的面积
- Precision

<span style="font-size:18px">**- 简单通用的算法**</span>

阈值调整（threshold moving）：将原本默认为0.5的阈值调整到 较少类别/（较少类别+较多类别）即可。使用现有的**集成学习**分类器，如随机森林或者xgboost，并调整分类阈值

<span style="font-size:18px">**- 对XGBoost模型数据的不平衡处理方法**</span>

通过正负样本的权重解决样本不均衡（一般分类中小样本量类别权重高，大样本类别权重低，再进行计算和建模

<span style="font-size:18px">**- 简单有效的方案**</span>

1. 不对数据进行过采样和欠采样，但使用现有的集成学习模型，如随机森林，XGBoost（lGBM）
2. 输出模型的预测概率，调整阈值得到最终结果
3. 选择合适的评估标准，如**precision**，Recall
4. 文本正则化中的任务是对测试集中的16个目标进行预测，训练集中的最大类别是PLAIN，为7353693，最小的类别为ADDRESS，为522。因此暂定PLAIN的权重为0.01，其余为1.（除去PLAIN，其余15个再做一次分类）



#### 2. 超参数优化（时间复杂度，空间复杂度）

如何选择合适的超参数？不同模型会有不同的最优超参数组合，找到这组最优超参数大家是根据经验或者随机的方法，来尝试。但是其是有可能用数学或者机器学习的模型来解决模型本身超参数的选择问题

<span style="font-size:18px">**背景**</span>

- 机器学习模型超参数调优一般被认为是一个黑盒优化问题，在调优过程中我们只能看到模型的输入与输出，不能获取模型训练过程中的梯度信息，也不能假设模型超参数和最终指标符合凸优化条件
- 模型训练代价大，时间，金钱成本

<span style="font-size:18px">**自动调参方法**</span>

Grid search（网格搜索），Random search（随机搜索），Genetic algorithm（遗传算法），Paticle Swarm Optimization（粒子群优化），Bayesian Optimization（贝叶斯优化），TPE，SMAC等

- **Genetic algorithm和PSO**是经典黑盒优化算法，归类为群体优化算法，不是特别适合模型超参数调优场景，因为其需要有足够多的初始样本点，并且优化效率不高**
- **Grid search**很容易理解与实现，但是遍历所有的超参数组合来找到其中最优化的方案，对于连续值还需要等间距采样。实际上这30种组合不一定取得全局最优解，而且计算量很大很容易组合爆炸，并不是一种高效的参数调优方法。
- **Random search**普遍被认为比Grid search效果好，虽然组合的超参数具有随机性，但是其出现效果可能特别差也可能特别好，在尝试次数和Grid search相同的情况下一般最值会更大，当然variance也更大但这不影响最终结果。

但是在计算机资源有限的情况下，Grid search与Random search不一定比建模工程师的经验要好

- **Bayesian Optimization**
  适用场景：

  （1）需要优化的function计算起来非常费时费力，比如上面提到的神经网络的超参问题，每一次训练神经网络都是燃烧好多GPU的
  （2）你要优化的function没有导数信息

#### 3. 可解释性工具

Xgboost相对于线性模型在进行预测时往往有更好的精度，但是同时也失去了线性模型的可解释性。所以Xgboost通常被认为是黑箱模型。
经典方法是使用**全局特征重要性评估**

2017年，Lundberg和Lee的[论文]( [A Unified Approach to Interpreting Model Predictions.pdf](../文献阅读/A Unified Approach to Interpreting Model Predictions.pdf) )提出了SHAP值这一广泛适用的方法用来解释各种模型（分类以及回归），其中最大的受益者莫过于之前难以被理解的黑箱模型，如boosting和神经网络模型。

1. 二分类，看下准确率，高的话
2. 集成XGBoost，LGB，随机森林
3. 可解释性，SHAP（SHAP值只能对特征进行分析）
4. 去掉PLAIN看下效果，ROC

### 后续改进

1. 将PLAIN的权值设置为0，训练结果：分数为0.98991
   将PLAIN去除不进行预测，实验结果无法得到官方分数，并且实验是通过上下文单词（context）来作为单位进行训练，若去除PLAIN无法训练。因此只能通过将权值设置为0，查看各个种类的预测准确率是否高，但是可以查看对训练集的效果

