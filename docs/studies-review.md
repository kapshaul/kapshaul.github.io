# Studies 내용 검토

검토일: 2026-09-27. 공개된 Studies 6편의 본문, 수식, 코드 예제와 연결된
실험 저장소를 대조했다. 원래 결과표의 수치와 그림 57개는 유지했다.
새로 계산한 값은 Discounted UCB 결과표의 산술평균뿐이다.

## 주요 수정

| 글 | 발견한 문제 | 반영한 수정 |
| --- | --- | --- |
| Attention | 유한 softmax를 정확한 선택·평균으로 해석, Gaussian scale의 제곱 기댓값 누락, 잘못된 텐서 전치·차원 설명 | 선택·평균의 극한 조건과 남은 확률질량 명시, E[λ²] = 1 + β, batch-first 예제와 선택적 padding mask |
| Word vectors | PPMI 명칭, 전체 GloVe 목적함수의 기울기에서 합 누락, t-SNE·analogy 결과에 대한 과도한 해석 | 확률·0-count 처리 정의, pair loss와 full gradient 구분, AG News 학습 결과와 pretrained word2vec 분석 분리 |
| LSTM / FSM | cell state를 AND로 설명, 유한 테스트로 임의 길이 일반화 단정, ERG를 무제한 재귀로 설명, POS 오답을 정답처럼 제시 | XOR 근사 설명, 검증 범위 명시, 유한상태 문법 설명과 올바른 문자열, POS padding loss 수정 예제와 오답 분석 |
| Sampling / beam search | beam width를 무작위 다양성으로 해석, 실제로 없는 batched inference 주장, greedy와 beam-1 결과 불일치 | 확률 필터·검색 구분, top-p 경계 토큰 처리, 실제 순차 beam 구현과 EOS 처리 부재 설명, 재현성 확인 항목 |
| Bandit comparison | UCB 식의 α 누락, normal의 표준편차를 분산으로 혼동, arm별 LinTS 샘플링을 표준 방식으로 설명, 선형 추정량을 비선형 MLE로 주장 | 실제 구현의 식·샘플링 위치 명시, 표준 알고리즘과 실험 변형 구분, squared-reward 목적함수와 식별성 설명 |
| Discounted UCB | 선택한 arm만 할인하는 구현을 시간 기반 할인으로 설명, γ≈1을 γ=1과 동일시, 출력값을 전체 horizon의 최종값으로 해석 | 모든 arm의 시간 할인 통계 예제, 구현 차이와 discount tradeoff 명시, 마지막 로그 지점과 전체 horizon 구분 |

Bandit 시뮬레이터는 0부터 100 iteration 간격으로 기록하고 마지막 기록값을
출력한다. Discounted UCB의 1,000회 설정에서는 iteration 900, 즉 901개
action까지의 값이다. 예전 표를 임의로 새 실험 결과로 바꾸지 않고 이 한계를
설명했다.

## 표현과 탐색

- 짧은 제목, 한 문단 요약, 세 개의 핵심 태그, 실제 검토일로 메타데이터 정리.
- Studies 목록에 NLP / Online Learning 필터 추가. 주제명도 검색 대상에 포함.
- 렌더링된 h2에서 목차와 anchor를 함께 생성하여 제목 변경 시 불일치 방지.
- 기존 HTML figure 모음을 semantic figure/figcaption으로 변환하고 설명적인 alt 제공.
- 반복 그래프는 details/summary로 접어서 표시. 본문 표는 긴 문장이 줄바꿈되도록 변경.
- 미공개 template의 이전 WordVector 메타데이터와 Hugo 전용 설정 제거.

## 대조한 코드 버전

검토 당시 각 저장소의 HEAD/branch commit이다. 외부 저장소는 수정하지 않았다.
OnlineLearning 두 행의 commit hash는 2026-09-27에 검토한 기록상의 snapshot이며,
링크는 아래 통합 이후 `main`에서 같은 내용을 가리키는 현재 경로다.

| 범위 | 확인한 revision |
| --- | --- |
| Attention | [6fcf262](https://github.com/kapshaul/NLP-attention.mechanism/tree/6fcf26272616a2128f88f7aec29db67bd8566954) |
| Word vectors | [95c3d51](https://github.com/kapshaul/NLP-WordVector/tree/95c3d51687f12c9f4ce5aa384c757e4a8f52f9ea) |
| LSTM / FSM | [c657616](https://github.com/kapshaul/NLP-finite.state.machine.RNN/tree/c657616e83180a89711f869dca116baab246992a) |
| Sampling | [53e54f2](https://github.com/kapshaul/NLP-sampling.search/tree/53e54f27ed66214fa1e68afda6ae86cc5daf288a) |
| Bandit comparison | 검토 당시 `bandits-comparison-analysis` `12f31f2` → 현재 [`main/docs/bandits-comparison.md`](https://github.com/kapshaul/OnlineLearning/blob/main/docs/bandits-comparison.md) |
| Discounted UCB | 검토 당시 `discountedUCB` `5066fcb` → 현재 [`main/docs/discounted-ucb.md`](https://github.com/kapshaul/OnlineLearning/blob/main/docs/discounted-ucb.md) |

알고리즘의 정의는 각 글에 연결한 Transformer, GloVe, t-SNE, nucleus sampling,
nonstationary UCB, generalized linear bandit 논문과 NumPy/PyTorch 공식 문서를
함께 대조했다.

### 2026-09-29 OnlineLearning branch 통합

OnlineLearning의 두 실험 branch가 `main` 하나로 통합되었다. 알고리즘 구현과
결과는 바뀌지 않았고, Bandit comparison과 Discounted UCB 글의 링크와 재현
안내만 새 경로에 맞게 수정했다.

| 이전 위치 | 현재 `main` 경로 |
| --- | --- |
| `bandits-comparison-analysis`: `Simulation.py`, `SimulationNonLinear.py`, `lib/` | 같은 파일 그대로 유지, 보고서는 `docs/bandits-comparison.md` |
| `discountedUCB`: `Simulation.py` | `SimulationDiscountedUCB.py` (byte 단위 동일) |
| `discountedUCB`: `lib/DiscountedUCBBandit.py` | 같은 경로로 변경 없이 복원 |
| `discountedUCB`: 보고서 | `docs/discounted-ucb.md` |

위의 검토일, 발견 사항과 아래 검증 결과는 2026-09-27 검토 기록 그대로다.
이번 통합에서 원본 결과표와 그림을 다시 생성한 것은 아니다.

## 검증

- `npm run lint` 통과.
- `npm run build` 통과: TypeScript 검사 및 정적 export 포함.
- `npm run verify` 통과: 공개 글 13편, 목록 3개, 내부 링크·asset 364개.
- 공개 Studies의 원본 그림 57개 보존, alt 누락 없음, 목차 44개 target 일치 및 중복 ID 없음.
- Python 코드 블록 5개 AST 문법 검사.
- Attention 선택 확률의 극한과 GloVe 전체 기울기의 finite-difference 검산.
- scalar LSTM의 길이 14인 모든 16,384개 이진 문자열에 대한 parity 분류 통과.
- 본문의 할인 통계 함수를 실행하여 직접 가중합 정의와 대조: γ = 0.1, 0.5, 0.9, 1.0.
- Discounted UCB 표의 평균 25.4 / 3.0 / 2.4 / 5.8 재계산.
- 브라우저에서 주제 필터, 검색, 결과 없음·초기화, 목차 링크, 접이식 그래프 확인.
- 390px 모바일 화면에서 본문 가로 넘침 및 KaTeX 오류 없음 확인.

PyTorch가 이 검증 환경에 없어 PyTorch 의존 예제는 실행하지 않았다.
학습·번역·텍스트 생성·bandit 실험을 재실행한 것은 아니며, 예전 결과와 수정된
예제를 본문에서 구분했다. 수치 검산은 Python 표준 라이브러리로 수행했다.
