from data_util.ImportDataUtil import *
from data_util.ProcessDataUtil import *

# 1 导入数据
# 2 将数据进行特征筛选，没有用的就废弃（y的走势都一样，影响不大）
read_file_obj = ImportDataUtil()
read_data = read_file_obj.read_csv_file('./data/insurance.csv', 'charges')

# 3 数据拆分
split_data = read_file_obj.trans_data(read_data, 0.3, 'x', 'y')
x_train = split_data.get('x_train')
y_train = split_data.get('y_train')
x_test = split_data.get('x_test')
y_test = split_data.get('y_test')

# 4 数据升维处理
poly_reg = ProcessDataUtil()
x_poly_data = poly_reg.upgrade_deep(x_train, x_test, 2)
x_poly_train = x_poly_data.get('x_train')
x_poly_test = x_poly_data.get('x_test')

# 5 进行模型预测
y_train_predict =  poly_reg.predict_by_linear_reg(x_poly_train, y_train)
y_test_predict = poly_reg.predict_by_linear_reg(x_poly_test, y_test)

estimate_data = poly_reg.estimate_data(y_train, y_train_predict)
print(estimate_data)
estimate_data_test = poly_reg.estimate_data(y_test, y_test_predict)
print(estimate_data_test)
print("========")

y_train_ridge = poly_reg.predict_by_ridge(x_poly_train, y_train)
y_test_ridge = poly_reg.predict_by_ridge(x_poly_test, y_test)

estimate_data_ridge_train = poly_reg.estimate_data(y_train, y_train_ridge)
estimate_data_ridge_test = poly_reg.estimate_data(y_test, y_test_ridge)
print(estimate_data_ridge_train, estimate_data_ridge_test)

gb_train_predict = poly_reg.booster_predict(x_poly_train, y_train)
gb_train_test = poly_reg.booster_predict(x_poly_test, y_test)

estimate_data_train_gb = poly_reg.estimate_data(y_train, gb_train_predict)
estimate_data_test_gb = poly_reg.estimate_data(y_test, gb_train_test)

print(estimate_data_train_gb, estimate_data_test_gb)

# 根据最后的结果来看 ，梯度提升的效果更好一点

