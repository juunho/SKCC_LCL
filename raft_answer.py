from copy import deepcopy
import json
from openai import OpenAI
import time
import jsonlines
import datetime
from collections import OrderedDict

# # preparing questions Batch File
# answer_template = {
#     "custom_id": None, # custom_id는 batch내에서 유일한 값을 가지도록 설정해야합니다.
#     "method": "POST", 
#     "url": "/v1/chat/completions",
#     "body": {"model": "gpt-4", #사용할 모델의 종류 입니다. 
#              "messages":[{"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."}],
#              "max_tokens": 1000 #모델이 대답으로 생성 가능한 토큰의 최대 길이입니다. 
#              }
#     }

# chunks = []
# titles = []
# questions = []
# with jsonlines.open("/workspace/RAFT/context_question_0620.jsonl") as f:        # 총 15842 개
#     for line in f.iter():
#         chunks.append(line["context"])
#         titles.append(line["title"])
#         questions.append(line["question"])


# batches = []
# for id in range(len(chunks)):
#     temp = deepcopy(answer_template)
#     question = questions[id]
#     chunk = chunks[id]

#     prompt = """
#         Question: {question}\nContext: {context}\n
#         Answer this question using the information given in the context above. Here is things to pay attention to: 
#         - First provide all responses in the same language as the question.
#         - Second, your response consists of a <REASONING> part for inferring the answer and <ANSWER> for the final answer.
#         - In the <REASONING>, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
#         You MUST begin your inferring with the tag "<REASONING>" and final answer with the tag "<ANSWER>, the answer should be succinct.".
#     """.format(question=question, context=str(chunk))

#     temp['custom_id'] = f'{id}'
#     temp['body']['messages'].append({"role": "user", "content": prompt})
#     batches.append(temp)

# with open('/workspace/RAFT/answer_input3.jsonl', 'w' , encoding='UTF-8') as file:
#     for item in batches:
#         json_string = json.dumps(item, ensure_ascii=False)        
#         file.write(json_string + '\n')



# client = OpenAI(api_key="sk-")
# # available only in version after openai==1.2.0
# batch_input_file = client.files.create(
#   file=open('/workspace/RAFT/answer_input3.jsonl', "rb"),
#   purpose="batch"
# )

# batch_input_file_id = batch_input_file.id

# batch_job=client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h", 
#     metadata={
#       "description": "nightly eval job"
#     }
# )

# print(f'Batch Job :')
# print(batch_job)
# print('-'*75)

# def execute_code(id: str):
#     batch_job = client.batches.retrieve(id)
#     if batch_job.status in ["validating", "in_progress", "finalizing", "cancelling"]:
#         flag = 0
#     elif batch_job.status in ["completed", "expired", "cancelled"]:
#         flag = 1
#     else:
#         flag = 99
#     return flag

# print('Batch Start !!!')
# start = time.time()

# flag = 0
# while flag == 0:
#     flag = execute_code(batch_job.id)
#     time.sleep(60)
#     print(f'Time : {str(datetime.timedelta(seconds=time.time()-start))} // Flag : {flag}')
#     if flag != 0 or (time.time()-start) >= 24 * 60 * 60:
#         print('Batch End !!!')
#         break
# print('-'*75)


###########################################################
###################### 결과확인 아래코드 ######################
###########################################################

 

client = OpenAI(api_key="sk-")
status = client.batches.retrieve("batch_fSAEzH5qKmLuiwibeG8ur8td").status
output_file_id = client.batches.retrieve("batch_fSAEzH5qKmLuiwibeG8ur8td").output_file_id
print(output_file_id)

result = client.files.content(output_file_id).content
result_file_name = "/workspace/RAFT/final_results_temp3.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)

# Loading data from saved file
results = []
with open(result_file_name, 'r', encoding='utf-8') as file:
    for line in file:
        json_object = json.loads(line.strip())
        results.append(json_object)


chunks = []
titles = []
questions = []
answers = {}
with jsonlines.open("/workspace/RAFT/context_question_0620.jsonl") as f:
    for line in f.iter():
        chunks.append(line["context"])
        titles.append(line["title"])
        questions.append(line["question"])

for res in results:
    answers[res['custom_id']] = res['response']['body']['choices'][0]['message']['content']
    
for cust_id in range(len(answers)):

    if cust_id != 2304 and cust_id != 2305 and cust_id != 2306 and cust_id != 2307 and cust_id != 2308 and cust_id != 2309 and cust_id != 33789 and cust_id != 33790 : 
        tmp_data = OrderedDict()
        tmp_data["title"] = titles[cust_id]
        tmp_data["context"] = chunks[cust_id]
        tmp_data["question"] = questions[cust_id]
        tmp_data["answer"] = answers[str(cust_id)]
        with open("/workspace/RAFT/RAFT_FINAL_0620.jsonl", "a", encoding="utf-8") as f:
            json.dump(tmp_data, f, ensure_ascii=False) # ensure_ascii로 한글이 깨지지 않게 저장
            f.write("\n")

