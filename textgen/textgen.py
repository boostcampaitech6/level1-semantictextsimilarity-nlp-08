from transformers import T5TokenizerFast, T5ForConditionalGeneration

import pandas as pd
from tqdm.auto import tqdm

def preprocessing(data):
    # 안쓰는 컬럼을 삭제합니다.
    data = data.drop(columns=['id'])

    # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
    try:
        targets = data['label'].values.tolist()
    except:
        targets = []
    # 텍스트 데이터를 전처리합니다    

    return data, targets

def text_generation(data):
    tokenizer = T5TokenizerFast.from_pretrained("paust/pko-chat-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("paust/pko-chat-t5-large").cuda()

    # # 5점짜리 prompt
    # prompt_tpl_rank5 = "\n문장을 조금 바꾸시오. 예시:\n''''''''''''''''''''''''''''''''''''''''''''''''''''문제:이국종교수님 지원해주세요\n정답:이국종 교수님을 지원해주세요\n''''''''''''''''''''''''''''''''''''''''''''''''''''\n문제:{text}\n\n정답:\n"
    # prompt_tpl_rank4 = "\n문장이 주어지면 주요 단어를 비슷한 단어로 바꾸시오. 예시:\n''''''''''''''''''''''''''''''''''''''''''''''''''''문제:곧.. 다시 만나요!!\n정답:곧.. 또 만나요!! \n''''''''''''''''''''''''''''''''''''''''''''''''''''\n문제:{text}\n\n정답:\n"
    # prompt_tpl_rank3 = "\n문장이 주어지면 주요 단어를 대체하시오. 예시\n:''''''''''''''''''''''''''''''''''''''''''''''''''''\n문제:외국인 무비자 입국페지요망\n정답:제주도 중국인 무비자 폐지 바랍니다.\n''''''''''''''''''''''''''''''''''''''''''''''''''''\n문제:{text}\n\n정답:\n"
    # prompt_tpl_rank2 = "\n문장이 주어지면 다른 의미가 되도록 비슷하게 바꾸시오. 예시\n:''''''''''''''''''''''''''''''''''''''''''''''''''''\n\n문제:허걱 ㅋㅋ 즐거운 대화였습니다.\n정답:눈물 살짝 훔치며 즐거운 이야기 나누었습니다! ㅋㅋㅋ \n```\n문제:{text}\n\n정답:\n"
    # prompt_tpl_rank1 = "\n문장이 주어지면 다른 의미가 되도록 바꾸시오. 예시\n:''''''''''''''''''''''''''''''''''''''''''''''''''''\n\n문제:국회의원 자격조건 요청합니다\n정답:국회의원 최저시급제로 전환 요청합니다 \n```\n문제:{text}\n\n정답:\n"
    # prompt_tpl_list = [prompt_tpl_rank1, prompt_tpl_rank2, prompt_tpl_rank3, prompt_tpl_rank4, prompt_tpl_rank5]
    # 문장 오타를 고치는 prompt
    prompt_tpl = "\n 문장이 주어지면 오타만 수정해주세요. 원 문장의 의미가 변경되면 안됩니다. \
                                예시:\n \
                                문장: 오마이가뜨지져스크롸이스트휏\n \
                                > 오 마이 갓 지저스 크라이스트.\n\n \
                                문장: 전 암만 찍어도 까만 하늘.. ㅠㅠ\n \
                                > 저는 아무리 찍어도 까만 하늘만 나와요.. ㅠㅠ\n\n \
                                문장: 이렇게 귀여운 쥐들은 처음이네요.ㅎㅎㅎ\n \
                                > 이렇게 귀여운 쥐들은 처음이네요.ㅎㅎㅎ\n \
                                문장:{text}\n> "

    sentence_1_list = []
    sentence_2_list = []
    generated_1 = []
    generated_2 = []
    count = 0
    for sentence_1, sentence_2 in tqdm(data[['sentence_1', 'sentence_2']].values, desc='generating', total=len(data)):
        #label = round(label)
        #prompt = prompt_tpl_list[label-1].format(text=sentence_1)
        prompt = prompt_tpl.format(text=sentence_1)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        logits = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.5,
            no_repeat_ngram_size=6,
            do_sample=True,
            num_return_sequences=1,
            )
        text = tokenizer.batch_decode(logits, skip_special_tokens=True)[0]
        sentence_1_list.append(sentence_1)
        generated_1.append(text)
        #prompt = prompt_tpl_list[label-1].format(text=sentence_2)
        prompt = prompt_tpl.format(text=sentence_2)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        logits = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=1,
            no_repeat_ngram_size=6,
            do_sample=True,
            num_return_sequences=1,
            )
        text = tokenizer.batch_decode(logits, skip_special_tokens=True)[0]
        sentence_2_list.append(sentence_2)
        generated_2.append(text)
        
        count += 1
        if count == 5:
            break

    return [sentence_1_list, generated_1, sentence_2_list, generated_2]


# read ../data/traincsv
train_df = pd.read_csv("../data/train.csv")
dev_df = pd.read_csv("../data/dev.csv")
test_df = pd.read_csv("../data/test.csv")

train_df, train_targets = preprocessing(train_df)
def_df, dev_targets = preprocessing(dev_df)
test_df, test_targets = preprocessing(test_df)

# Generate text
generated = text_generation(train_df)
generated_dev = text_generation(dev_df)
generated_test = text_generation(test_df)

#print all
print(generated)
print(generated_dev)
print(generated_test)

# Save generated text
#generated_df = pd.DataFrame({'sentence_1': generated[0], 'generated_1': generated[1], 'sentence_2': generated[2], 'generated_2': generated[3], 'label': generated[4]})
#generated_df.to_csv("../data/generated.csv", index=False)

train_df['sentence_1'] = generated[1]
train_df['sentence_2'] = generated[3]
train_df.to_csv("../data/converted_train.csv", index=False)

dev_df['sentence_1'] = generated_dev[1]
dev_df['sentence_2'] = generated_dev[3]
dev_df.to_csv("../data/converted_dev.csv", index=False)

test_df['sentence_1'] = generated_test[1]
test_df['sentence_2'] = generated_test[3]
test_df.to_csv("../data/converted_test.csv", index=False)

print("Done!")