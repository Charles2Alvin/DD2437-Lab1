# 项目部署：https://marlous.github.io/2019/04/11/Python-%E9%A1%B9%E7%9B%AE%E7%9A%84%E5%88%9B%E5%BB%BA%E5%BC%80%E5%8F%91%E3%80%81%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2%E9%97%AE%E9%A2%98/
from prepare_data import DataProducer

if __name__ == '__main__':
    n = 100
    patterns, targets = DataProducer.produce(n, True)
    print(type(targets))
