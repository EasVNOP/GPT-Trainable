from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
text_generation_zh  = pipeline(task=Tasks.text_generation, model='baichuan-inc/baichuan-7B', device_map='auto',model_revision='v1.0.5')
text_generation_zh._model_prepare = True
result_zh = text_generation_zh('今天天气是真的', min_length=10, max_length=512, num_beams=3,temperature=0.8,do_sample=False, early_stopping=True,top_k=50,top_p=0.8, repetition_penalty=1.2, length_penalty=1.2, no_repeat_ngram_size=6)
print(result_zh)