from transformers import T5TokenizerFast, T5ForConditionalGeneration

import pandas as pd
from tqdm.auto import tqdm

def preprocessing(data):
    # 안쓰는 컬럼을 삭제합니다.
    data = data.drop(columns=['source','id', 'binary-label'])

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

    # 5점짜리 prompt
    prompt_tpl_rank5 = "문제가 주어지면 문제와 동일한 의미를 가지지만 주요 단어를 교체하시오.\n ```\n예시: 정형식판사 감사요청\nanswer: 정형식 판사 감사요청 \n```\n문제:{text}\n\nanswer:\n"
    prompt_tpl_rank4 = "문제가 주어지면 문제의 단어를 교체하여 비슷한 다른 문장을 작성하시오.\n ```\n예시: 정말 간절히 정부의 도움이 필요합니다.\nanswer: 정부의 도움이 절실히 필요합니다. \n```\n문제:{text}\n\nanswer:\n"
    prompt_tpl_rank3 = "문제가 주어지면 비스무리한 문장으로 대체하시오.\n ```\n예시:\n \n문제: 오늘이 최종회여서 정말x 아쉬워요..\nanswer: 오늘이 마지막이라니 정말 아쉽다는... \n```\n문제:{text}\n\nanswer:\n"
    prompt_tpl_rank2 = "문제가 주어지면 문제와 비슷해보이지만 서로 유사하지 않은 문장으로 단어를 교체하시오.\n ```\n예시: 부트캠프 프로그램 얘기부터 추천 도메인 얘기까지 순식간에 시간이 흘러가서 너무 아쉬웠습니다 ㅎㅎ\nanswer: 각자 업무와 부트캠프 얘기부터 앞으로 진로 얘기까지 나누다보니 한시간 훌쩍 넘게 수다를 떨었네요. \n```\n문제:{text}\n\nanswer:\n"
    prompt_tpl_rank1 = "문제가 주어지면 문장과 비슷해보이지만 서로 의미가 상관없도록 단어를 교체하시오.\n ```\n예시: 둘 다 낯가리면서도 수다스러움을 감출 수 없는 캐릭터라는 것을 깨닫고 ㅋㅋ\nanswer: 재밌는 컨셉을 써도 숨길 수 없는 퀄리티ㅠㅠㅠ \n```\n문제:{text}\n\nanswer:\n"
    prompt_tpl_list = [prompt_tpl_rank1, prompt_tpl_rank2, prompt_tpl_rank3, prompt_tpl_rank4, prompt_tpl_rank5]

    generated_1 = []
    generated_2 = []
    label_list = []
    count=0
    for sentence_1, sentence_2, label in tqdm(data[['sentence_1', 'sentence_2', 'label']].values, desc='generating', total=len(data)):
        label = round(label)
        prompt = prompt_tpl_list[label-1].format(text=sentence_1)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        logits = model.generate(
            input_ids,
            max_new_tokens=len(sentence_1),
            temperature=0.5,
            no_repeat_ngram_size=6,
            do_sample=True,
            num_return_sequences=1,
            )
        text = tokenizer.batch_decode(logits, skip_special_tokens=True)[0]
        generated_1.append(text)
        prompt = prompt_tpl_list[label-1].format(text=sentence_2)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        logits = model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.5,
            no_repeat_ngram_size=6,
            do_sample=True,
            num_return_sequences=1,
            )
        generated_2.append(text)
        label_list.append(label)
        
        count += 1
        if count == 10:
            break

    return [generated_1, generated_2, label_list]


# read ../data/traincsv
train_df = pd.read_csv("../data/train.csv")
train_df, train_targets = preprocessing(train_df)

# Generate text
generated = text_generation(train_df)

# Save generated text
generated_df = pd.DataFrame({'sentence_1': generated[0], 'sentence_2': generated[1], 'label': generated[2][:10]})
generated_df.to_csv("../data/generated.csv", index=False)