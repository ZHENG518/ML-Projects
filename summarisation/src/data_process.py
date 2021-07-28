import pandas as pd
import re
import jieba

def load_train():
    train_df = pd.read_csv('../data/AutoMaster_TrainSet.csv')
    # 去除没有答案的行
    train_df.dropna(how='any', subset=['Report'], inplace=True)
    # 去除没有输入的行
    train_df.dropna(how='all', subset=['Question','Dialogue'], inplace=True)
    # 填补Question Dialogue的空值
    train_df.fillna(value={'Question':'', 'Dialogue':''}, inplace=True)

    return train_df

def load_test():
    test_df = pd.read_csv('../data/AutoMaster_TestSet.csv')
    return test_df

def text_process(line):
    """
    将一行的Question，Dialogue合并并清洗后返回
    """
    texts = [' '.join(jieba.lcut(line['Question']))]
    # 提取对话中技师说的内容
    for dialogue in line['Dialogue'].split('|'):
        if '技师说：' in dialogue:
            dialogue = re.sub('技师说：|\[语音]|\[图片]', '', dialogue)
            if dialogue:
                texts.append(' '.join(jieba.lcut(dialogue)))
    return ' '.join(texts)

def train_valid_split():
    train_df = pd.read_csv('../data/train.csv')
    train_df = train_df.sample(frac=1)
    train_size = int(train_df.shape[0]*0.7)
    train = train_df[:train_size]
    valid = train_df[train_size:]
    train.to_csv('./data/train.csv', index=False)
    valid.to_csv('./data/valid.csv', index=False)

if __name__ == '__main__':
    train_df_raw = load_train()
    train_text = train_df_raw.apply(text_process, axis=1)
    reports = [' '.join(jieba.lcut(report)) for report in train_df_raw.Report]
    train_df = pd.DataFrame({'input':train_text, 'report':reports})
    train_df.to_csv('./data/train.csv', index=False)

    test_df_raw = load_test()
    test_text = test_df_raw.apply(text_process, axis=1)
    test_id = test_df_raw.QID
    test_df = pd.DataFrame({'QID': test_id, 'input': test_text})
    test_df.to_csv('./data/test.csv', index=False)

    # train_valid_split()