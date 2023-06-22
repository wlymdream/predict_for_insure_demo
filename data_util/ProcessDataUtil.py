# 对数据进行预处理
import numpy as np
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble  import GradientBoostingRegressor

class ProcessDataUtil:

    '''
        对数据进行升维处理 解决欠拟合问题
        x_train : 训练集数据
        x_test : 测试集数据
    '''
    def upgrade_deep(self, x_train, x_test, degree_size):
        #不考虑截距项
        poly_obj = PolynomialFeatures(degree=degree_size, include_bias=False)
        x_train = poly_obj.fit_transform(x_train)
        x_test = poly_obj.fit_transform(x_test)
        return {'x_train': x_train, 'x_test': x_test}

    '''
        利用回归去计算预测值 由于线性回归，我们是假设误差是服从正态分布的，那么我们将y值也变成正态分布的 
    '''
    def predict_by_linear_reg(self, x, y):
        linear_reg = LinearRegression()
        linear_reg.fit(x, np.log(y))
        y_predict = linear_reg.predict(x)
        return y_predict

    '''
        岭回归 考虑惩罚项
        
    '''
    def predict_by_ridge(self, x, y):
        ridge_reg = Ridge()
        ridge_reg.fit(x, np.log(y))
        y_predict = ridge_reg.predict(x)
        return y_predict

    '''
        梯度提升回归
    '''
    def booster_predict(self, x, y):
        gb_reg = GradientBoostingRegressor()
        gb_reg.fit(x, np.log(y))
        y_predict = gb_reg.predict(x)
        return y_predict

    '''
        对模型进行评估
    '''
    def estimate_data(self, y_true, y_predict):
        # squared True返回mse false返回rmse 就是mse开根号 由于我们在计算预测值的时候进行了log 所以这里给他还原，用exp
        data = mean_squared_error(y_true, np.exp(y_predict), squared=True)
        return data


