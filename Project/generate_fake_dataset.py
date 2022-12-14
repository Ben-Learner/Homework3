import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
torch.manual_seed(1234)
class My_dataset(Dataset):
    def __init__(self, raw_path):
        """
        Args:
            raw_path: 原始数据路径
        """
        self.feature = []
        self.label = []
        accident_name = os.listdir(raw_path)
        accident_sample = [self._process_item(os.path.join(raw_path, accident), accident) for accident in accident_name]
        self.label = pd.Categorical(self.label).codes
        assert len(self.label) == len(self.feature)
    def _minmax_norm(self, df_input):
        """
        Args:
            df_input: 每个样本的df文件
        Returns:归一化的结果
        """
        for i in list(df_input.columns):
            Max = np.max(df_input[i])
            df_input[i] = df_input[i] / (Max + 1e-6) #分母加平滑
        return df_input.values

    def _process_item(self, accident_path, accident):
        """
        Args:
            accident_path: 每个事故得路径
            accident: 事故的名字
        Returns:None

        """
        for sample in os.listdir(accident_path):
            self.label.append(accident)
            sample = pd.read_csv(os.path.join(accident_path, sample)).iloc[:150, 1:97] #取前1500s数据,后面有部分数据多出三列
            sample = self._minmax_norm(sample)
            # sample = list(itertools.chain.from_iterable(sample)) #将数据展开为1维
            self.feature.append(sample)

    def __getitem__(self, item):
        """
        Args:
            item: 样本索引

        Returns:单个样本

        """
        x = self.feature[item]
        y = self.label[item]
        # return {'x': x, 'y':y}
        return (x,y)
    def __len__(self):
        """
        Returns:样本长度
        """
        return len(self.label)

if __name__ == "__main__":
    raw_path = './dataset'
    accident_data = My_dataset(raw_path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=accident_data, lengths=[0.8, 0.2])
    train_loader, test_loader = DataLoader(train_dataset, batch_size=16, shuffle=True), DataLoader(test_dataset, batch_size=16)
    # for batch_idx, samples in enumerate(train_loader):
    #     print(batch_idx, samples)
    #     break
    for inputs, labels in train_loader:
        print(inputs.shape, labels.shape)
        break





