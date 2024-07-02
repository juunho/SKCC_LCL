from preprocessing import Preprocessing
from gsdmm import MovieGroupProcess
import numpy as np

def compute_V(texts):
    V = set()
    for text in texts:
        for word in text:
            V.add(word)
    return len(V)

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print("\nCluster %s : %s"%(cluster,sort_dicts))


if __name__ == "__main__":
    # Example input text
    input_text = "본 장에서는 일반 대출 신청자 및 씬파일러 데이터 세트에서 베이스라인 모델과 TeGCN 모델 간 성능 비교를 수행했다. 이 때, 모델의 일반화 가능성을 검증하기 위해 K-Fold 교차검증을 수행하였다. 교차검증은 총 5회 실시하였으며, 각 Fold의 평균을 통해 도출한 실험 결과는 표 5와 같다. TeGCN 모델은 일반 대출 신청자 데이터에 대하여 5개의 머신러닝 및 딥러닝 기반 베이스라인 모델보다 성능이 뛰어난 것으로 나타났다. AUC 값과, K-S 통계량 두 평가 지표에서 모두 TeGCN 모델이 우수한 것으로 나타났으며, 이는 그래프 정보를 활용한 신용 평가 모형이 채무 불이행 예측이 유의미한 것을 나타낸다. 또한, 금융 이력이 부족한 씬파일러 데이터 세트에서 역시 베이스라인 모델 대비 높은 성능을 보였다. 특히, 일부 베이스라인 모델들이 씬파일러 예측 시 성능이 떨어진데 반해, 본 모델은 비교적 안정적인 성능을 유지했다. 전체 데이터 세트와 마찬가지로 씬파일러 데이터에서도 기준 모델보다 TeGCN 모델의 성능이 뛰어난 것으로 나타났다. 마지막으로, TeGCN은 Tab Transformer 대신 MLP 레이어를 추가한 GCN 모델 대비 뛰어난 성능을 기록했다. 이는 Tab Transformer를 활용한 범주형 변수 임베딩이 채무 불이행 예측 성능 향상에 유의미한 도움이 된다는 것을 의미한다. 추가적으로, 타겟 클래스 균형에 대한 모델의 영향과 성능을 확인하기 위하여 소수 클래스와 다수 클래스의 비율을 1:1로 맞춰 모델에 대한 Sub-Sample 테스트를 아래 표 6과 같이 진행하였다. 그 결과, 클래스 균형의 상황에서도 모델은 기존 모델 대비 준수한 성능을 보였다. 본 연구는 모델 학습에 활용된 23개의 핵심 변수들의 중요도를 분석하였다. 중요도 평가를 위하여 Feature Permutation 방식을 적용하였다. 이 방법은 특정 변수의 값을 임의로 섞어, 기존 출력과 변경된 출력 사이의 차이로 해당 변수의 중요도를 측정한다. 만약 해당 변수 값을 변경해도 모델 출력에 크게 영향을 미치지 않는다면, 그 변수는 상대적으로 중요하지 않다는 해석이 가능하다. 그러나 큰 차이가 발생한다면, 해당 변수는 모델에서 핵심적인 역할을 수행한다고 볼 수 있다. Feature Permutation 방식을 통해 확인한 변수 중요도는 그림 3과 같다. ‘분할 납부금(installment)’과 ‘대출 금액(loan_amount)’등의 주요 대출 관련 변수는 채무 불이행 예측에 결정적인 영향을 주는 것으로 나타났다. 그 외에도 ‘리볼빙 서비스 이용률 (revlo util)’과 ‘모기지 계좌 수(mort_acc)’ 그리고 ‘신용계좌 수(open_acc)’와 같은 변수들 또한 중요한 역할을 하는 것으로 평가되었다."

    # Preprocessing step
    processor = Preprocessing(input_text)
    preprocessed_texts = processor.preprocess()

    # Initialize MovieGroupProcess
    mgp = MovieGroupProcess(K=20, n_iters=200, alpha=0.1, beta=0.1)
    
    # Compute vocabulary size
    V = compute_V(preprocessed_texts)
    # Fit MovieGroupProcess with preprocessed texts
    y = mgp.fit(preprocessed_texts, V)

    # Print the result
    print("----------------------------------- RESULT -----------------------------------")
    
    # Number of documents per topic
    doc_count = np.array(mgp.cluster_doc_count)
    print('Number of documents per topic :', doc_count)
    
    # Topics sorted by the number of documents they are allocated to
    top_index = doc_count.argsort()[-10:][::-1]
    print('\nMost important clusters (by number of docs inside):', top_index)
    
    # Show the top 10 words in term frequency for each cluster
    top_words(mgp.cluster_word_distribution, top_index, 10)