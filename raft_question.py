from copy import deepcopy
import json
from openai import OpenAI
import time
import jsonlines
import datetime
from collections import OrderedDict
import tiktoken

# preparing questions Batch File
question_template = {
    "custom_id": None, # custom_id는 batch내에서 유일한 값을 가지도록 설정해야합니다.
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4", #사용할 모델의 종류 입니다. 
             "messages":[],
             "max_tokens": 1000 #모델이 대답으로 생성 가능한 토큰의 최대 길이입니다. 
             }
    }

def question_size(context):

    if len(context) > 1500:
        return 10
    elif len(context) > 1000:
        return 8
    elif len(context) > 500:
        return 6
    elif len(context) > 200:
        return 4
    else : 
        return 0
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

chunks = []
titles = []
with jsonlines.open("/workspace/RAFT/merged_all.jsonl") as f:        # 총 15842 개
    for line in f.iter():
        chunks.append(line["content"])
        titles.append(line["title"])

chunks = chunks[6990:]
titles = titles[6990:]

batches = []
for id in range(len(chunks)):
    temp = deepcopy(question_template)
    chunk = chunks[id]

    tokens = num_tokens_from_string(chunk, "cl100k_base")
    
    if tokens < 8300 :
        qs_size = question_size(chunk)
        temp['custom_id'] = f'{id}'
        temp['body']['messages'].append({"role": "system", "content": "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'" % (qs_size)})

        temp['body']['messages'].append({"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only {} korean and {} english questions in your response.".format(qs_size//2 , qs_size//2)})

        temp['body']['messages'].append({"role": "user", "content": str(chunk)})
    
        batches.append(temp)
    
    else : 
        print("index : {} is too long (tokens : {})".format(id+6991,tokens))


with open('/workspace/RAFT/question_input.jsonl', 'w' , encoding='UTF-8') as file:
    for item in batches:
        json_string = json.dumps(item, ensure_ascii=False)        
        file.write(json_string + '\n')



client = OpenAI(api_key="sk-proj-cbXiyApkdizUeuRpEJFvT3BlbkFJ5OUxLvS9TbjRNiEmWKYK")
# available only in version after openai==1.2.0
batch_input_file = client.files.create(
  file=open('/workspace/RAFT/question_input.jsonl', "rb"),
  purpose="batch"
)

batch_input_file_id = batch_input_file.id

batch_job=client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h", 
    metadata={
      "description": "nightly eval job"
    }
)

print(f'Batch Job :')
print(batch_job)
print('-'*75)

def execute_code(id: str):
    batch_job = client.batches.retrieve(id)
    if batch_job.status in ["validating", "in_progress", "finalizing", "cancelling"]:
        flag = 0
    elif batch_job.status in ["completed", "expired", "cancelled"]:
        flag = 1
    else:
        flag = 99
    return flag

print('Batch Start !!!')
start = time.time()

flag = 0
while flag == 0:
    flag = execute_code(batch_job.id)
    time.sleep(60)
    print(f'Time : {str(datetime.timedelta(seconds=time.time()-start))} // Flag : {flag}')
    if flag != 0 or (time.time()-start) >= 24 * 60 * 60:
        print('Batch End !!!')
        break
print('-'*75)


# ###########################################################
# ###################### 결과확인 아래코드 ######################
# ###########################################################

 

# client = OpenAI(api_key="sk-proj-cbXiyApkdizUeuRpEJFvT3BlbkFJ5OUxLvS9TbjRNiEmWKYK")
# status = client.batches.retrieve("batch_CgkMQcBxBiKHvVcdHOe8fdDs").status
# output_file_id = client.batches.retrieve("batch_CgkMQcBxBiKHvVcdHOe8fdDs").output_file_id
# print(output_file_id)

# result = client.files.content(output_file_id).content
# result_file_name = "/workspace/RAFT/final_questinos_results_temp.jsonl"

# with open(result_file_name, 'wb') as file:
#     file.write(result)

# # Loading data from saved file
# results = []
# with open(result_file_name, 'r', encoding='utf-8') as file:
#     for line in file:
#         json_object = json.loads(line.strip())
#         results.append(json_object)


# chunks = []
# titles = []
# with jsonlines.open("/workspace/RAFT/merged_all.jsonl") as f:        # 총 15842 개
#     for line in f.iter():
#         chunks.append(line["content"])
#         titles.append(line["title"])

# chunks = chunks[6990:]
# titles = titles[6990:]

# def strip_str(s: str) -> str:
#     """
#     Helper function for helping format strings returned by GPT-4.
#     """
#     l, r = 0, len(s)-1
#     beg_found = False
#     for i in range(len(s)):
#         if s[i].isalpha():
#             if not beg_found:
#                 l = i
#                 beg_found = True
#             else:
#                 r = i 
#     r += 2
#     return s[l:min(r, len(s))]

# questions = {}

# for res in results:
#     questions[res['custom_id']] = res['response']['body']['choices'][0]['message']['content']
    
# for cust_id in range(len(questions)):
#     if cust_id != 1951 and cust_id != 9153 and cust_id != 6153 and cust_id != 11042  : # table 길어진것 빼고
        
#         queries =  questions[str(cust_id)].split('\n')
#         queries = [strip_str(q) for q in queries]
#         queries = [q for q in queries if any(c.isalpha() for c in q)]

#         for q in queries :
#             tmp_data = OrderedDict()
#             tmp_data["title"] = titles[cust_id]
#             tmp_data["context"] = chunks[cust_id]
#             tmp_data["question"] = q
        
#             if tmp_data["question"] != '':
#                 with open("/workspace/RAFT/context_question_0619_2.jsonl", "a", encoding="utf-8") as f:
#                     json.dump(tmp_data, f, ensure_ascii=False) # ensure_ascii로 한글이 깨지지 않게 저장
#                     f.write("\n")
#     else:
#         print(cust_id)
#         a = chunks[cust_id]


