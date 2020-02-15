# 동화창작 web 서비스

## 만들게 된 이유

Naver AI hackerton burning day 참가 하려고 하였던 아이디어를 구현한것이다. 

1차 코딩테스트와 기획안 제출 부분에서 코딩테스트는 무난하게 넘어갔지만

기획서에서 문제가 있는건지 탈락하게 되어서 개인프로젝트로 구현하였다.



## 구현 예정 (변경 가능)

front  : 무료 웹템플릿을 가져와 변형시킨 것으로 메인부분과 파일 재생 페이지 두개로 분류하여 구성

server : apache + php + mongodb 로 APM 구성하였으며 확장성 유동성을 위해 docker 로 구성하고 모델은 k8s를 통해서 배포가 가능하도록 만들 예정

model : 

GPT2 774 model 을 기반으로 text genetor 를 구성하고

음성합성을 하여 따로 특정인의 목소리로  최종 음성 출력 할수 있도록 한다.









 