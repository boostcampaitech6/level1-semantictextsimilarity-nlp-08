import pandas as pd

class TextPreprocesser():
    def __init__(self, load_path: str, save_path: str):
        """TextPreprocesser의 인스턴스 생성

        Args:
            load_path (str): preprocessing할 csv파일을 가져올 경로 -> '~~.csv'
            save_path (str): preprocessing한 csv파일을 저장할 경로 -> '~~.csv'
        """
        self.load_path = load_path
        self.save_path = save_path

    def preprocessing(self):
        """실제 preprocessing을 진행하는 함수
        """
        # load csv file from self.load_path
        df = pd.read_csv(self.load_path)

        n = len(df)
        for idx in range(0, n):
            if df['label'][idx] == 0:
                df.loc[idx+n+1] = ['boostcamp-sts-v1-df-' + str(idx+n+1), 
                                    df['source'][idx],
                                    df['sentence_1'][idx], 
                                    df['sentence_1'][idx],
                                    5.0,
                                    1.0]
                df.loc[idx+n+2] = ['boostcamp-sts-v1-df-' + str(idx+n+2), 
                                    df['source'][idx],
                                    df['sentence_2'][idx], 
                                    df['sentence_2'][idx],
                                    5.0,
                                    1.0]
            elif df['label'][idx] == 5:
                df.loc[idx+n+1] = ['boostcamp-sts-v1-df-' + str(idx+n+1), 
                                    df['source'][idx],
                                    df['sentence_1'][idx], 
                                    df['sentence_1'][idx][::-1],
                                    0.0,
                                    0.0]
                df.loc[idx+n+2] = ['boostcamp-sts-v1-df-' + str(idx+n+2), 
                                    df['source'][idx],
                                    df['sentence_2'][idx], 
                                    df['sentence_2'][idx][::-1],
                                    0.0,
                                    0.0]
                
        m = len(df)
        for idx in df.index:
            df.loc[idx+m+1] = ['boostcamp-sts-v1-df-' + str(idx+m+1),
                                    df['source'][idx],
                                    df['sentence_2'][idx], 
                                    df['sentence_1'][idx],
                                    df['label'][idx],
                                    df['binary-label'][idx]]
        
        # save csv file as self.save_path
        df.to_csv(self.save_path, index=False)