import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

class ImportDataUtil:

    '''
        读取文件
        file_dictionary : 文件路径
        y_name 哪一行数据作为y列
    '''
    def read_csv_file(self, file_dictionary, y_name):
        data = pd.read_csv(file_dictionary, sep = ',')
        # 丢掉y列
        x = data.drop(y_name, axis=1)
        # 丢掉对y影响都一样的列，规律都差不多，我们去把他剔除，进行人工降噪处理
        x_data = x.drop(['region', 'sex'], axis=1)
        # 对一部分数据进行离散化 dataFrame中的支持自定义的方法
        x_dispersed = x_data.apply(self.dispersed_data, axis=1, args=(30, 0))
        y = data[y_name]
        # one-hot编码 处理字符串数据 将值的类型拆分成多个列 例如 sex 拆分成 sex.female和sex.male
        x_trans_data = pd.get_dummies(x_dispersed)
        # 空值填充

        # shade 阴影 利用画图的方式可以去掉一部分影响不大的数据
        # sns.kdeplot(data.loc[data.sex == 'male', 'charges'], legend=True)
        # sns.kdeplot(data.loc[data.sex == 'female', 'charges'], legend=True)
        x_trans_data.fillna(0, inplace=True)
        y.fillna(0, inplace=True)
        return {'x': x_trans_data, 'y': np.log(y)}

    '''
        数据中的bmi >= 30我们算作肥胖 小于30算作不肥胖，我们将连续型的数值变成离散化
        datas : 类型为dataFrame 这里用x的数据
        bmi_num : bmi的标准值
        children_num : 有没有孩子
    '''
    def dispersed_data(self, datas, bmi_num, children_num):
        datas['bmi'] = 'over' if datas['bmi'] >= bmi_num else 'under'
        datas['children'] = 'no' if datas['children'] == children_num else 'yes'
        return datas
    '''
        拆分数据
        data : 要拆分的数据 data :{x:fjdsak, y:fhjsadik} 元组类型
        split_size : 
        x_key : data中的x的key 
        y_key : data中y的key 
    '''
    def trans_data(self, data, split_size, x_key, y_key):
        x = data.get(x_key)
        y = data.get(y_key)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size)
        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}





