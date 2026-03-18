# Memory Decay System — 연구 일지

## 2026-03-17 (Day 1)

### 프로젝트 기획 및 설계

**배경:** autoresearch (karpathy/autoresearch) 메타실험 경험을 바탕으로, LLM 에이전트가 파라미터를 자율 탐색하는 연구 패러다임을 응용한 새로운 연구를 기획.

**연구 주제:** 인간과 유사한 메모리 감쇠 시스템의 계산적 모델링

**핵심 아이디어:**
- 각 기억에 활성도(activation score)를 부여하고, 시간에 따라 감쇠 함수로 점수 하락
- 새 정보가 들어올 때 기존 기억과의 연관성을 통해 re-activation 발생
- Impact (감정적 중요도)가 높은 기억은 감쇠가 느림
- 의미 기억(fact)과 에피소드 기억(episode)을 구분하여 각각 다른 감쇠 파라미터 적용

### 설계 결정 사항

| 항목 | 결정 | 이유 |
|------|------|------|
| 감쇠 함수 | 지수 + 멱함수 비교 | Ebbinghaus 망각 곡선 (멱함수) vs 표준 지수 감쇠 |
| 연관성 판단 | 그래프 + 임베딩 하이브리드 | LLM 매번 호출은 비효율, 임베딩만으로는 문맥 부족 |
| 시간 단위 | 임의 tick | 순수 연구이므로 추상화, 나중에 스케일 변환 가능 |
| 기억 단위 | fact + episode 혼합 | 신경과학의 의미/에피소드 기억 구분 반영 |
| 평가 방식 | 5개 메트릭 + composite score | 단일 메트릭 게이밍 방지 (autoresearch 교훈) |
| LLM 제공자 | OpenAI (gpt-4o-mini) | Anthropic 코스트 만료 → 전환, 비용 ~$0.015/run |

### 구현된 컴포넌트

| 모듈 | 파일 | 설명 |
|------|------|------|
| MemoryGraph | `graph.py` | NetworkX DiGraph, sentence-transformers 임베딩, 코사인 유사도 검색, re-activation cascade |
| DecayEngine | `decay.py` | 지수/멱함수 감쇠, fact/episode별 파라미터, impact modifier |
| Evaluator | `evaluator.py` | recall_rate, precision_rate, activation-recall 상관계수, fact/episode 차이, 곡선 부드러움 |
| SyntheticDataGenerator | `data_gen.py` | OpenAI API로 합성 기억 데이터 생성 (hub-and-leaf topology) |
| AutoImprover | `auto_improver.py` | LLM 자율 파라미터 최적화 (minimal/default/expert guidance) |
| Main | `main.py` | CLI + 통합 시뮬레이션 파이프라인 |

**테스트:** 40개 단위/통합 테스트 전부 통과

### 발견된 이슈 및 수정 (Feedback)

#### 1. Impact modifier가 감쇠를 역전시킴
- **문제:** 초기 설계 `A(t) = A₀ * e^(-λt) * (1 + α * impact)`에서 `(1 + 0.5 * 0.9) = 1.45`로 곱해져서 감쇠 후에도 활성도가 오히려 증가함. recall이 tick 100까지 1.000으로 고정.
- **해결:** impact를 곱셈이 아닌 나눗셈으로 적용 `λ_eff = λ / (1 + α * impact)`. 높은 impact → 낮은 유효 λ → 느린 감쇠. 활성도 상한을 1.0으로 clamping.

#### 2. LLM이 요청한 메모리 개수를 정확히 맞추지 않음
- **문제:** 50개 요청 → 40개만 생성. LLM이 JSON 배열 크기를 임의로 조정.
- **해결:** retry 로직 추가. 80% 이상 생성되면 성공, 미달 시 재시도 (최대 3회). 프롬프트에 "정확히 N개" 강조.

#### 3. Re-activation cap이 너무 높음
- **문제:** 활성도가 2.0까지 허용되어 감쇠 의미가 퇴색.
- **해결:** cap을 1.0으로 조정. 인간의 기억은 완벽에 가까워도 100%가 아님.

#### 4. AutoImprover의 .env 로딩 로직 불안정
- **문제:** Path 기반 .env 파일을 자동 탐색하는 로직이 복잡하고 깨지기 쉬움.
- **해결:** 환경변수 `OPENAI_API_KEY`만 사용하도록 단순화. `.env` 파일은 사용자가 직접 관리.

#### 5. 데이터 생성 API 호출 타임아웃
- **문제:** WSL2 환경에서 OpenAI API 응답이 간헐적으로 느려져 프로세스 타임아웃 발생.
- **상태:** 미해결. 백그라운드 실행 + 긴 타임아웃으로 회피 중. 네트워크 환경 의존적.

### autoresearch 실험에서 얻은 교훈 반영

| 교훈 | memory-decay에서의 적용 |
|------|----------------------|
| 단일 메트릭은 게이밍당함 | 5개 메트릭 + 가중 composite score |
| val_bpb=0.001은 암기임 | recall > 0.95 지속 시 memorization flag |
| SSSL attention 게이밍 | 파라미터 범위 clamping (validate_params) |
| misleading prompt 테스트 | guidance level 3단계 (minimal/default/expert) |

### 다음 단계 (TODO)

- [ ] 데이터 생성 안정화 (타임아웃 문제 해결)
- [ ] 실제 실험 실행: exponential vs power_law 비교
- [ ] auto-improvement 루프 정상 동작 확인
- [ ] 결과 시각화 (matplotlib: forgetting curve, recall over time)
- [ ] guidance level별 성능 비교 실험
- [ ] impact ablation study
- [ ] 최종 리포트 작성

## 2026-03-18 (Day 2)

### Human Calibration Layer 추가

실제 인간 복습 로그를 바로 기존 synthetic 그래프 실험에 섞지 않고, 별도 `human calibration` 레이어로 분리했다.

핵심 결정:

- 인간 로그는 `fact` 파라미터 보정에만 사용
- synthetic graph benchmark는 외적 타당성 검증 용도로 유지
- `episode`는 직접 학습하지 않고 제약 기반 외삽으로 유지

추가된 컴포넌트:

| 모듈 | 파일 | 설명 |
|------|------|------|
| Human Data | `human_data.py` | Duolingo/Anki 스타일 이벤트 정규화, leakage-safe user split |
| Human Eval | `human_eval.py` | review event replay, sigmoid observation model, `nll`/`brier`/`ece` |
| Human Optimizer | `human_optimizer.py` | `fact` 파라미터 random search |
| Human Runner | `human_runner.py` | calibration artifact 생성 (`best_params.json`, `metrics.json`, `trials.json`) |

연구 범위 제한:

- 실제 인간 리뷰 로그는 `lambda_fact`, `stability_weight`, `stability_decay`, `reinforcement_gain_direct` 계열 보정에 우선 사용
- `lambda_episode`는 인간 로그에서 직접 학습하지 않음
- association cascade 관련 파라미터는 여전히 synthetic benchmark에서 해석해야 함

검증:

- 새 human calibration 테스트 12개 통과
- 기존 `runner`, `simulation`, `cache_builder` 회귀 테스트 17개 통과

### 스모크 테스트용 fixture

- `data/human_reviews_smoke.jsonl` 추가
- 목적: 전체 실데이터 이전에 fact-only calibration 경로를 빠르게 확인하는 최소 입력

### 저장소

- **GitHub:** https://github.com/tmdgusya/memory-decay (private)
- **로컬:** `~/.openclaw/workspace/memory-decay/`

## 2026-03-18 (Day 2, Session 2)

### Auto-Research Loop: exp_0111–0175 (65 experiments)

#### 요약

총 65개 실험을 실행하여 **overall score 0.4099 → 0.4261** (+3.9%) 달성.

| 메트릭 | 이전 최고 (exp_0082) | 새 최고 (exp_0163) | 변화 |
|--------|---------------------|-------------------|------|
| **Overall** | 0.4099 | **0.4261** | +0.0162 |
| Retrieval | 0.2968 | 0.2966 | ~0 |
| Plausibility | 0.6737 | **0.7282** | +0.0545 |
| Correlation | 0.1115 | **0.2931** | +0.1816 |
| Smoothness | 0.9146 | 0.9146 | 0 |
| Recall | 0.3902 | 0.3902 | 0 |
| Precision | 0.0788 | 0.0780 | ~0 |

#### 핵심 발견

1. **Retrieval score는 구조적으로 고정되어 있다**
   - Recall = similarity_recall_rate (0.390) — 활성도 임계값이 아닌 임베딩 유사도 top-5 검색이 병목
   - 65개 실험 모두 동일한 retrieval (0.2966) — floor, lambda, alpha, 감쇠 형태와 무관
   - Precision도 동일 (0.078) — 검색 결과의 관련성은 임베딩 품질에 의존

2. **개선 여지는 plausibility뿐이다**
   - Plausibility = 0.3×correlation + 0.7×smoothness
   - Smoothness는 이미 0.9146으로 포화 → 유일한 레버는 **correlation**
   - Correlation: activation과 recall success 간 Pearson 상관계수

3. **Base floor 발견 (exp_0114–0132)**
   - `base_floor` 파라미터 도입: 모든 메모리에 최소 활성도 하한 부여
   - base_floor=0.79에서 최적 (floor > consolidation_threshold=0.7 → 진동 효과)
   - Correlation 0.1115 → 0.2795 (+150%)

4. **Impact-dependent damping 발견 (exp_0161–0163)**
   - Consolidation phase에서 impact에 비례한 감쇠 속도 조절
   - `damping = cd_base + cd_impact × (1 - impact)` — 저 impact → 빠른 감쇠, 고 impact → 느린 감쇠
   - 최적: cd_base=0.1, cd_impact=1.0 → correlation 0.2931

#### 새로운 최고 decay function (exp_0163)

```python
# 핵심 변경: impact-dependent consolidation damping
damping = cd_base + cd_impact * (1.0 - impact)  # 0.1 ~ 1.1
rate = lam * damping / combined                  # 고 impact → 느린 감쇠
```

기존 고정 damping (0.4) 대신, impact에 따라 0.1~1.1 범위로 가변.

#### 실패한 접근들

| 접근 | 결과 | 실패 원인 |
|------|------|----------|
| Sigmoidal impact gating | 0.274 | Recall 0.134로 급락 — 너무 공격적 감쇠 |
| Power-law floor (impact²) | 0.316 | 같은 이유 — 선택적 floor가 recall 손실 |
| Inverse-square decay | 0.101 | 대재앙 — 거의 모든 메모리 소실 |
| Alpha 변경 (1.5–3.0) | 0.400–0.415 | Retrieval 불변, plausibility 미미한 변화 |
| Reinforcement params (10종) | 0.4099 (전부 동일) | 강화 이벤트가 점수에 영향 없음 |
| 단일 phase (floor-only) | 0.410 | 두 phase 조합이 더 효과적 |

#### 구조적 한계 분석

현재 시스템에서 decay function으로 개선 가능한 범위는 거의 소진됨:

- **Recall**: 임베딩 유사도 검색 top-5에 의해 결정. Decay function과 무관.
- **Precision**: 같은 이유. 검색 품질은 임베딩 모델에 의존.
- **Smoothness**: 0.9146으로 포화. 추가 개선 여지 극소.
- **Correlation**: 0.2931. Impact가 실제 retrievability를 예측하는 정도에 의해 상한 결정.

Correlation의 이론적 상한: impact와 similarity-based retrievability 간 상관관계에 의존. Impact 분포와 임베딩 품질이 고정된 상태에서 activation spread 최대화만 가능.

#### Escalation 기록

Retrieval 개선을 위해서는 allowed search surface 밖의 변경이 필요:
- 임베딩 모델 교체 또는 top_k 증가 (evaluator.py)
- 데이터셋의 recall_query 품질 개선 (dataset)
- 시뮬레이션 길이 변경 (runner.py)
