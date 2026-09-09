import inspect

class BaseConfig:
    def __init__(self) -> None:
        """递归初始化所有成员类，忽略以“__”开头的名称（内置方法）。"""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # 遍历所有属性名称
        for key in dir(obj):
            # 忽略内置属性
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # 获取对应的属性对象
            var =  getattr(obj, key)
            # 检查该属性是否为类
            if inspect.isclass(var):
                # 实例化该类
                i_var = var()
                # 将属性从类类型替换为对应实例
                setattr(obj, key, i_var)
                # 递归初始化该属性的成员
                BaseConfig.init_member_classes(i_var)
