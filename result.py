import transformers
import torch

result = "./results/checkpoint-20"

pipeline = transformers.pipeline("text-generation", model=result, model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation":"flash_attention_2"}, device_map="auto")
a = pipeline("벽에 걸린 괘종시계가 8시를 알리며 둔한 종소리를 내기 시작했다. 그리고 그와 동시에 문이 열리고 그곳으로 한 남자가 걸어 들어왔다. 전신을 둘러싼 검은 코드 안쪽에 얼핏 비치는 복장은 틀림없는 가톨릭 사제복이다. 비록 술을 금하지는 않지만 경건한 가톨릭의 사제가 직접 바를 찾아온다는 것은 그다지 흔치 않은 일이었다. 더구나 은발의 외국인 젊은이라면."
             , max_length=1000)
print(a)


# import torch
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# result = "./results/checkpoint-20"

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=False,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     result
#     , quantization_config=quant_config
#     , torch_dtype=torch.bfloat16
#     , attn_implementation="flash_attention_2"
#     , local_files_only=True)
# model.config.use_cache = False

# tokenizer = AutoTokenizer.from_pretrained(result)

# tokenizer.pad_token = "[PAD]"
# # tokenizer.pad_token = tokenizer.bos_token
# tokenizer.padding_side = "left"

# prompt = "Hello, my llama is cute"
# # inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# generated_ids = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.7)
# # generated_ids = model.generate(**inputs)
# outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# print(outputs)