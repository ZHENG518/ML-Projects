# 百度AI studio比赛-汽车大师问答摘要与推理

### Data

- 

### Tips

- LSTM,GRU解决了梯度消失，没有解决梯度爆炸，应将learning rate设小

- `scores.data.masked_fill_(padding_mask, -float('inf'))`将<PAD>位置的attention score设置为0

- `CrossEntropyLoss(reduction='sum', ignore_index=padding_idx)`忽略<PAD>的loss
- schedule sampling：随着epoch增加，teacher forcing rate减小
- `torch.cuda.empty_cache()`用来清理gpu缓存

### Train

- train:10000, valid:10000, epoch:30, learning_reate:0.001, linear decay scheduled sampling
  - Valid_loss: 0.9510 Valid_acc: 0.0147, Train_loss: 0.8090 Train_acc: 0.0177

- train:10000, valid:10000, epoch:20, learning_reate:0.001, linear decay scheduled sampling
  - Valid_loss: 1.3942 Valid_acc: 0.0028, Train_loss: 1.9559 Train_acc: 0.0082
  - 因为没有将<PAD>的loss忽略

