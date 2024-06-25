import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from bonito import Bonito

bonito = Bonito("BatsResearch/bonito-v1")

from datasets import Dataset
from vllm import SamplingParams
from transformers import set_seed

set_seed(2)

from pprint import pprint

unannotated_paragraph = """1. “Confidential Information”, whenever used in this Agreement, shall mean any data, document, specification and other information or material, that is delivered or disclosed by UNHCR to the Recipient in any form whatsoever, whether orally, visually in writing or otherwise (including computerized form), and that, at the time of disclosure to the Recipient, is designated as confidential."""

pprint(unannotated_paragraph)
def convert_to_dataset(text):
    dataset = Dataset.from_list([{"input": text}])
    return dataset

def bonito_gen(unannotated_paragraph, task_of_type):
    sampling_params = SamplingParams(max_tokens=512, top_p=0.95, temperature=0.5, n=1)
    synthetic_dataset = bonito.generate_tasks(
        convert_to_dataset(unannotated_paragraph),
        context_col="input",
        task_type=task_of_type,
        sampling_params=sampling_params,
    )

    pprint("----Generated Instructions----")
    print(f'Input: {synthetic_dataset[0]["input"]}')
    print(f'Output: {synthetic_dataset[0]["output"]}')


set_seed(0)
sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.7, n=1)
synthetic_dataset = bonito.generate_tasks(
    convert_to_dataset(unannotated_paragraph),
    context_col="input",
    task_type="mcqa",  # changed
    sampling_params=sampling_params,
)
pprint("----Generated Instructions----")
pprint(f'Input: {synthetic_dataset[0]["input"]}')
pprint(f'Output: {synthetic_dataset[0]["output"]}')



from datasets import load_dataset
import re
finqa = load_dataset("PatronusAI/financebench")

# 0 62 121
unannotated_paragraph = re.sub("\n+\s+", "\n", finqa['train']['evidence_text'][121])
unannotated_paragraph = """일본은 노인보건제도 시행 과정에서 발생한 문제점을 개선하기 위해 2008년 4월부터 후기고령자의료제도를 시행하고 있다 일본에서 그동안 시행해 왔던 노인보건제도는 중앙정부 시 군 구 부담금 및 건강보험료로 재원이 조달되고 있었으나 고령화가 진행되면서 현역 세대의 보험료가 지속적으로 증가하게 되었다 또한 국민건강보험의 경우 시군구 간의 보험료 부담 격차가 최대 다섯배까지 발생하게 되었으며 보험료 격차를 줄이기 위해 시 군 구 단위의 운영주체를 광역연합으로 개선할 필요성이 제기되었다 이에 일본은 새로운 고령자 의료제도에 대한 필요성을 공통적으로 인식함에 따라 75세 이상 후기고령자의 심신의 특성 및 생활 실태 등을 고려하여 후기 고령자의료보험을 독립된 제도로 시행하였다 동시에 65세부터 74세까지의 전기고령자에 대해서는 퇴직자가 국민건강보험에 대량 가입함으로 인해 보험자 간의 의료비 부담에 불균형이 발생하는 현상을 해결하기 위한 재정조정제도를 시행하고 있다 보험료는 가입자 전원으로부터 징수하며 징수관련 업무는 시 군 구가 담당하고 재정운영은 전 시 군 구가 가입하는 시도 단위의 광역연합 2010년 기준 47개 이 담당한다 보험료는 광역연합별로 결정되는데 연금이 일정액 월 15 000엔 이상인 후기고령자의 경우 보험료는 공적연금에서 원천징수한다 보험료는 정액부분 50 과 소득비례부분 50 으로 구성되며 정액보험료는 동경지역과 오사카 지역의 경우 각각 5만 엔 4만 엔 2012 수준이다 보험요율은 시도별로 차이가 있을 수 있지만 광역연합 내에서는 균일하게 설정하는 것을 원칙으로 하여 지역 간 보험료 격차를 개선하였다 동 건강보험 재정은 75세 이상 고령자의 보험료 10 국고 50 정부 및 지방정부 각 건강보험자로부터 지원금 40 로 운영한다 환자 본인부담은 10 를 원칙으로 하나 고소득자의 경우 일반인과 동일하게 30 의 본인 부담을 하고 있다."""
unannotated_paragraph = """2 미국 가 건강보험체계 미국의 의료보장체계를 타 국가의 의료시스템과 비교해 볼 때 가장 특징적인 점은 공적건강보험의 역할 비중이 증가하고는 있지만 민영건강보험의 역할이 매우 높다는 것이다 즉 미국의 경우 전국민을 대상으로 한 공적건강보험제도는 존재하지 않으며 노인과 장애자등 일부 한정된 자를 대상으로 한 Medicare와 일부 저소득층을 대상으로 한 메디케이드 Medicaid 제도가 운영되고 있다 미국의 경우 전 국민의 64 가 민간보험에 가입하고 있는데 이 중 대부분은 고용주지원 직장건강보험 Employer Sponsored Health Benefits 에 가입되어 있다 2010년 기준으로 직장건강보험 가입자는 전체 민영건강보험 가입자의 86 4 를 차지하고 있으며 보험종류 및 가입옵션에 따라 고용주와 근로자가 비용을 달리 분담하고 있다 한편 고용주가 보험을 제공하지 않거나 실업자여서 개인적으로 민영건강보험에 가입할 경우 직장건강보험을 통해 가입할 때보다 보험료가 크게 증가하게 된다 """
unannotated_paragraph = """2014년 호텔과 카지노 기업의 사이버보험 가입 증가율은 69 로 가장 높았으며 그 다음으로는 교육기관 58 공공서비스 기업 47 유통 기업 43 제조 기업 35 서비스기업 27 금융기관 18 통신 미디어와 IT 기업 8 의료기관 7 순으로 나타났다 사이버 공격이 발생할 경우 호텔과 카지노 기업은 영업활동 중단 회사 이미지 훼손 고객 감소 등의 경제적 손실이 다른 산업에 비해 매우 크기 때문에 이러한 위험을 감소시키기 위해 사이버보험에 많이 가입 한 것으로 조사 되었다 사이버보험은 개인정보 유출에 따른 손해배상금 및 방어 비용 뿐만 아니라 회사가 관리하는 기업정보 유출에 따른 법률상 손해배상금 정보보안 실패로 인한 정보 훼손에 따른 법률상 손해배상금까지 종합적으로 보장한다 또한 사이버 사고에 보다 더 체계적인 대응을 지원하기 위해 사고원인을 밝히기 위한 조사 법률 자문 신원 모니터링 비용 등의 위기관리 비용까지 보장하며 사이버보험의 보장범위와 보험료는 해당 산업 서비스 형태 데이터 위험 노출 수준 네트워크 보안 수준 개인정보보호 정책 매출 등에 근거하여 산출한다 일반적으로 100만 달러 보장 시 보험료는 대기업은 2만 5천 5만 달러 중소기업은 1만 5천 2만 달러 수준이며 보험사는 사이버보험 가입자에게 비밀번호 수시 변경 데이터의 분류 보관 정기적인 스트레스 테스트 일정 수립 등 안전수칙 지도 및 관리를 실시한다 미국 국토안보부는 사이버 공격으로 인한 데이터 유출 영업 방해 네트워크 손실 등의 피해를 줄이기 위해 기업들의 사이버보험 가입을 장려하고 있다 2015년 뮌헨 리 Munich Re 사가 실시한 조사에 따르면 기업들의 비즈니스 리스크 관리자들에게 사이버 위협에 대처하기 위해 회사에 어떠한 대처방안을 취하고 있는지 조사한 결과 70 의 응답자가 사이버 보험을 구매했다고 응답하였으며 나머지 30 는 보험을 구입하지 않은 것으로 응답하였다 보험을 구매한 기업 중 28 6 는 작년과 동일한 보험 보장 수준을 유지하고 있다고 응답하였으며 35 7 는 작년보다 더 보험 혜택을 늘렸다고 응답하였고 나머지 35 7 는 올해 처음으로 보험을 구매한 기업이었다 보험을 구매하지 않은 기업의 44 는 사이버보험이 너무 복잡하다는 이유로 구매하지 않고 있으며 34 는 충분한 위협 인식을 하지 못하고 있는 것으로 나타났다 """
print(unannotated_paragraph)
finqa['train']['question'][1]
finqa['train']['answer'][0]

bonito_gen(unannotated_paragraph, "nli")