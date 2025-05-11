#这个是数据加载内层，设置具体加载路径
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = r'E:\变化检测\变化检测数据集\LEVIR-CD+\tiles(256)'
        elif data_name == 'CDD':
            self.label_transform = "norm"
            self.root_dir = r'E:\变化检测\变化检测数据集\CDD\tiles(256)'
        elif data_name == 'WHUCD':
            self.label_transform = "norm"
            self.root_dir = r'E:\变化检测\变化检测数据集\WHU-CD\tiles(256)'

        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

