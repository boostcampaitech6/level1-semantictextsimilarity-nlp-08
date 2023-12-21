import pandas as pd
from pororo import Pororo

def data_aug(df, pg):
    '''
    
    '''
    # 다운 샘플링
    for s in df['source'].unique():
        if 'sampled' in s:
            filter_df = df[(df['source'] == s) & (df['label'] == 0.0 )]
            sampling_df = filter_df.sample(len(filter_df) // 2)
            
            df = df.drop(sampling_df.index)
            
            sampling_df["sentence_2"] = sampling_df["sentence_1"]
            sampling_df["label"] = 5.0
            sampling_df["binary-label"] = 1.0
            
            df = pd.concat([df, sampling_df])
            
    filter1 = df[(df['label'] >= 1.5) & (df['label'] <= 3.0)]
    f1_df = filter1.sample(len(filter1) // 2).copy()

    f1_df["sentence_1"] = filter1["sentence_1"].apply(pg)
    
    augmented_df = pd.concat([df, f1_df])
    
    return augmented_df

def data_swap(df):
    '''
    
    '''
    # 다운 샘플링
    for s in df['source'].unique():
        if 'sampled' in s:
            filter_df = df[(df['source'] == s) & (df['label'] == 0.0 )]
            sampling_df = filter_df.sample(len(filter_df) // 2)
            
            df = df.drop(sampling_df.index)
            
            sampling_df["sentence_2"] = sampling_df["sentence_1"]
            sampling_df["label"] = 5.0
            sampling_df["binary-label"] = 1.0
            
            df = pd.concat([df, sampling_df])
            
    filter1 = df[(df['label'] >= 1.5) & (df['label'] <= 3.0)]
    f1_df = filter1.sample(len(filter1) - len(filter1) // 3).copy()

    f1_df["sentence_1"], f1_df["sentence_2"] = filter1["sentence_2"], filter1["sentence_1"]
    
    augmented_df = pd.concat([df, f1_df])
    
    return augmented_df

if __name__ == '__main__':
    train_path = '../data/train.csv'
    
    df = pd.read_csv(train_path)
    df_sampling = df.copy()
           
    pg = Pororo(task="pg", lang="ko")
    df_sampling = data_aug(df_sampling, pg)

    df_sampling.to_csv('train_aug.csv', index=False)