# Learning Collabo Lab in SK C&amp;C
**사업向 sLLM에서 Service Delivery 시 겪고 있는 문제를 개선하기 위한 AutoRAG for Domain(가제)을 구현하였습니다.**

본 프로젝트는 SK C&C Learning Collabo Lab의 지원을 받아 진행하였습니다.

</br>


## 개요
~~~
Partner: SK C&C
Date : 2024.04 ~ 2024.10
Team: G.AI Engieering 1팀 배준호 매니저, 신현준 매니저, 이용택 매니저, 한예지 매니저
~~~

</br>

## 문제 정의
> 사업向 sLLM에는 (1) 도메인 특화된 데이터의 부족, (2) 도메인 별 수치 데이터 핸들링 이라는 두 가지 중요한 문제가 존재합니다.
>
> 이는 통신과 금융 도메인 고객사 대상의 sLLM 사업에서 가장 큰 허들로 여겨지고 있습니다.
>
> **💡 따라서, 기존 RAG의 파이프라인을 개선한 'AutoRAG for Domain(가제)'를 통해 Service Delivery를 혁신하고자 합니다.**


</br>

## 접근 방법

<p align="center">
  <img src="https://github.com/juunho/Completed_Projects/assets/81394769/a5b693a0-1f5f-47a0-bc6e-afd3b0b4550a" width="80%">
</p>

</br>

- sLLM을 이용한 도메인 별 데이터 전처리

- RAFT, Bonito를 통한 Synthetic Data

- RAG Evaluation
