class Path(object):
    '''
    视频数据集路径及切分后的帧保存路径
    '''
    @staticmethod
    def db_dir(database):
        if database == 'ff':
            # 原始的包含类别标签视频数据集路径
            # root_dir = r'/root/autodl-tmp/datasets'
            root_dir = r"D:\pythonProject1\datasets"

            # 数据集的视频切分帧后保存的路径
            # out_dir = r'/root/autodl-tmp/datasets/ff'
            out_dir = r"D:\pythonProject1\datasets\ff"
            return root_dir, out_dir
        else:
            print("Database {} not available".format(database))
            raise NotImplementedError



