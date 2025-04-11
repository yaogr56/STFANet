from dataloaders.dataloader import MyDataset
from torch.utils.data import DataLoader

dataset = 'ff'

train_dataloader = DataLoader(MyDataset(dataset=dataset, split='train', clip_len=32, preprocess=False),
                                  batch_size=8, shuffle=True, num_workers=8)

print(train_dataloader)
# print(train_dataloader.shape)
for _ in train_dataloader:
    print(_)