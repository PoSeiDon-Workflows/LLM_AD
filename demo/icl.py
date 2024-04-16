# %% [markdown]
# # In-context learning
# * dataset: 1000genome
# * model: Mistral-7b-v0.1

# %%
from transformers import pipeline, AutoTokenizer
import torch
torch.manual_seed(0)

# %%
# load dataset and model
ckp = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(ckp)
pipe = pipeline('text-generation',
                model=ckp,
                tokenizer=ckp,
                device="cuda:0"
                # torch_dtype=torch.bfloat16,
                )

# %% [markdown]
# ## ICL with zero-shot learning

# %%
prompt1 = """
You are a system administration bot.
Your task is to assess a job description with couple of features into one of the following categories:
Normal
Abnormal

You will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.

A single job has six features, including "wms delay", "queue delay", "runtime", "post script delay", "stage in delay", and "stage out delay".

Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 987 sec, post script delay 5 sec, stage in delay 65 sec, and stage out delay 4 sec
Category:
"""
sequences = pipe(prompt1,
                 max_length=100,
                 #  max_new_tokens=8,
                 do_sample=True)
for seq in sequences:
    print(seq['generated_text'])

# %% [markdown]
# ## ICL with few-shot learning

# %%
prompt2 = """
You are a system administration bot.
Your task is to assess a job description with couple of features into one of the following categories:
Normal
Abnormal

You will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.

A single job has six features, including "wms delay", "queue delay", "runtime", "post script delay", "stage in delay", and "stage out delay".
For the normal job,
the mean and std of wms delay is 5.25 and 0.91 seconds, respectively.
the mean and std of queue delay is 228.3 and 506.9 seconds, respectively.
the mean and std of run time is 970.9 and 819.3 seconds, respectively.
the mean and std of post script delay is 5.0 and 0.2 seconds, respectively.
the mean and std of stage in delay is 69.7 and 134.7 seconds, respectively.
and the mean and std of stage out delay is 4.8 and 8.8 seconds, respectively.

For the abnormal job,
the mean and std of wms delay is 5.24 and 0.6 seconds, respectively.
the mean and std of queue delay is 208.3 and 602.9 seconds, respectively.
the mean and std of run time is 1775.6 and 1360.0 seconds, respectively.
the mean and std of post script delay is 5.0 and 0.2 seconds, respectively.
the mean and std of stage in delay is 182.6 and 287.9 seconds, respectively.
and the mean and std of stage out delay is 6.1 and 20.5 seconds, respectively.

### Example ###
Instruct: Job has wms delay 1 sec, queue delay 4 sec, runtime 6 sec, post script delay 5 sec, stage in delay 0 sec, and stage out delay 0 sec
Category: Normal
Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 3268 sec, post script delay 5 sec, stage in delay 65 sec, and stage out delay 1 sec
Category: Abnormal
Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 832 sec, post script delay 5 sec, stage in delay 57 sec, and stage out delay 4 sec
Category: Normal
Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 987 sec, post script delay 5 sec, stage in delay 65 sec, and stage out delay 4 sec
Category:
"""
sequences = pipe(prompt2,
                 max_length=100,
                 #  max_new_tokens=8,
                 do_sample=True)
for seq in sequences:
    print(seq['generated_text'])

# %% [markdown]
# ## ICL with Chain-of-Thoughts

prompt3 = """
You are a system administration bot.
Your task is to assess a job description with couple of features into one of the following categories:
Normal
Abnormal

A single job has six features, including "wms delay", "queue delay", "runtime", "post script delay", "stage in delay", and "stage out delay".
For the normal job,
the mean and std of wms delay is 5.25 and 0.91 seconds, respectively.
the mean and std of queue delay is 228.3 and 506.9 seconds, respectively.
the mean and std of run time is 970.9 and 819.3 seconds, respectively.
the mean and std of post script delay is 5.0 and 0.2 seconds, respectively.
the mean and std of stage in delay is 69.7 and 134.7 seconds, respectively.
and the mean and std of stage out delay is 4.8 and 8.8 seconds, respectively.

For the abnormal job,
the mean and std of wms delay is 5.24 and 0.6 seconds, respectively.
the mean and std of queue delay is 208.3 and 602.9 seconds, respectively.
the mean and std of run time is 1775.6 and 1360.0 seconds, respectively.
the mean and std of post script delay is 5.0 and 0.2 seconds, respectively.
the mean and std of stage in delay is 182.6 and 287.9 seconds, respectively.
and the mean and std of stage out delay is 6.1 and 20.5 seconds, respectively.

### Example ###
Instruct: Job has wms delay 1 sec, queue delay 4 sec, runtime 6 sec, post script delay 5 sec, stage in delay 0 sec, and stage out delay 0 sec
Category: Normal
Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 3268 sec, post script delay 5 sec, stage in delay 65 sec, and stage out delay 1 sec
Category: Abnormal
Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 832 sec, post script delay 5 sec, stage in delay 57 sec, and stage out delay 4 sec
Category: Normal
Instruct: Job has wms delay 5 sec, queue delay 20 sec, runtime 987 sec, post script delay 5 sec, stage in delay 65 sec, and stage out delay 4 sec
Category:

Please think about it step-by-step.
"""
sequences = pipe(prompt3,
                 max_length=500,
                 #  max_new_tokens=8,
                 do_sample=True)
for seq in sequences:
    print(seq['generated_text'])

# response
"""
Sure, here's the step-by-step reasoning:

1. Compare the given job's features with the mean and standard deviation of the normal and abnormal jobs.
2. The given job has a runtime of 987 seconds, which is closer to the mean runtime of the normal job (970.9 seconds) than the mean runtime of the abnormal job (1775.6 seconds).
3. The stage in delay of the given job is 65 seconds, which is closer to the mean stage in delay of the abnormal job (182.6 seconds) than the mean stage in delay of the normal job (69.7 seconds).
4. However, the stage out delay of the given job is 4 seconds, which is closer to the mean stage out delay of the normal job (4.8 seconds) than the mean stage out delay of the abnormal job (6.1 seconds).
5. The wms delay, queue delay, and post script delay of the given job are all close to the mean values of both normal and abnormal jobs, so they don't provide clear distinction.
6. Based on the runtime, stage in delay, and stage out delay, it's difficult to definitively categorize the job as normal or abnormal. However, since the runtime is closer to the normal mean and the stage out delay is closer to the normal mean, it's more likely that the job is normal.

Therefore, the category is likely Normal, but it's a close call.
"""
