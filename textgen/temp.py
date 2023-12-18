from transformers import T5TokenizerFast, T5ForConditionalGeneration
tokenizer = T5TokenizerFast.from_pretrained("paust/pko-chat-t5-large")
model = T5ForConditionalGeneration.from_pretrained("paust/pko-chat-t5-large")

# 5점짜리 prompt
prompt_tpl = "문장을 주어지면 문장과 매우 비슷한 문장이지만 주요 단어를 대체하시오.\n 예시:\n '''\n문제: 정형식판사 감사요청\nanswer: 정형식 판사 감사요청 \n'''\n문제:{text}\n\nanswer:\n"
# 3점짜리 prompt
question = "스릴도있고 반전도 있고 여느 한국영화 쓰레기들하고는 차원이 다르네요~"
#prompt_tpl = "문장이 주어지면 문장과 매우 비슷한 문장이지만 주요 단어를 대체하시오.\n 예시:\n '''\n문제: 오늘이 최종회여서 정말x 아쉬워요..\nanswer: 오늘이 마지막이라니 정말 아쉽다는... \n'''\n문제:{text}\n\nanswer:\n"

# 문장 생성기
prompt_tpl = "짧은 문장을 만들고, 이와 비슷한 문장을 만드시오.\n 예시:\n '''\n오늘이 최종회여서 정말x 아쉬워요.., 오늘이 마지막이라니 정말 아쉽다는...\n'''"

prompt = prompt_tpl.format()#text="스릴도있고 반전도 있고 여느 한국영화 쓰레기들하고는 차원이 다르네요~")
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
logits = model.generate(
    input_ids,
    max_new_tokens=1024,
    temperature=0.5,
    no_repeat_ngram_size=6,
    do_sample=True,
    num_return_sequences=1,
)
text = tokenizer.batch_decode(logits, skip_special_tokens=True)[0]
print(f"input: {question}, generated: {text}")
