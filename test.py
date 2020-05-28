from scipy.stats import expon, norm,uniform # 指数分布の機能を提供するオブジェクト

s = expon(scale = 1.0 / (1e-100 + 5))
print(s)
