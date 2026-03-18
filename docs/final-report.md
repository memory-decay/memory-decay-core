# 인간 기억에서 영감을 받은 AI 메모리 감쇠 모델:
# 강화(reinforcement)를 포함한 지수 감쇠와 거듭제곱 법칙의 재설계

**Human-Memory-Inspired AI Memory Decay Model:  
Reinforcement-Aware Redesign of Exponential and Power-Law Forgetting**

---

## 초록

본 문서는 기존의 `activation-only` 감쇠 모델을 `activation + stability` 이중 상태 모델로 재설계한 연구 드래프트이다. 이전 버전은 시간 경과에 따른 망각과 연관 기억의 일시적 활성화는 다루었지만, 반복 상기에 따라 기억이 장기적으로 단단해지는 reinforcement 효과를 충분히 표현하지 못했다. 또한 단일 activation threshold(0.3)에 지나치게 의존하고, 종합 점수와 실제 recall 해석이 충돌하는 문제가 있었다.  

재설계된 시스템은 (1) 기억의 즉시 회상 가능성을 나타내는 `activation_score`와 (2) 반복 상기를 통해 축적되는 `stability_score`를 분리하고, impact와 stability가 함께 미래 감쇠율을 늦추도록 모델링한다. 평가 체계는 단일 threshold 기반 점수 대신 `[0.2, 0.3, 0.4, 0.5]` threshold grid에서의 recall/precision 평균을 주 지표로 삼고, activation-recall correlation 및 forgetting-curve smoothness를 plausibility 지표로 분리한다.  

이 드래프트의 핵심 기여는 인간 기억을 직접 모사한다고 주장하는 대신, **인간 기억에서 영감을 받은 AI memory system** 관점에서 감쇠와 강화의 상호작용을 더 일관되게 모델링하는 데 있다. 다만 현재 실험은 여전히 합성 한국어 메모리 데이터에 기반하며, 수치 결과와 그림은 새 스크립트로 재산출되어야 한다.

**키워드:** memory decay, reinforcement, forgetting curve, graph memory, activation, stability, threshold sweep, auto-improvement

---

## 1. 문제 정의

이 프로젝트의 목표는 인간 기억 이론을 문자 그대로 재현하는 것이 아니라, AI 에이전트용 장기 메모리 시스템에서 유용한 `기억 약화 + 반복 상기 강화` 메커니즘을 계산적으로 구현하는 것이다. 기존 버전은 다음 한계를 가졌다.

1. 감쇠는 모델링했지만 reinforcement는 activation boost 수준에 머물렀다.
2. 평가가 단일 threshold에 고정되어 결과 해석이 민감했다.
3. 종합 점수(composite score)가 실제 retrieval 성능과 다른 승자를 만들 수 있었다.
4. “human memory”라는 표현에 비해 합성 데이터 기반 검증 범위가 좁았다.

이번 재설계는 위 네 문제를 동시에 다루는 것을 목표로 한다.

---

## 2. 시스템 재설계

### 2.1 상태 변수

각 memory node는 다음 핵심 상태를 가진다.

- `activation_score`: 현재 회상 가능성에 가까운 단기 활성도
- `stability_score`: 반복 상기를 통해 축적되는 장기 안정성
- `impact`: 기억의 중요도
- `retrieval_count`: direct recall 횟수
- `last_reinforced_tick`: 마지막 reinforcement 시점

기존 `last_activated_tick`은 호환성 목적으로 유지하되, 의미적으로는 deprecated 상태로 간주한다.

### 2.2 감쇠 식

재설계된 기본 감쇠식은 impact와 stability를 모두 사용한다.

**Exponential decay**

$$
\lambda_{\mathrm{eff}} = \frac{\lambda_{type}}{(1 + \alpha \cdot impact)(1 + \rho \cdot stability)}
$$

$$
A_{t+1} = A_t \cdot e^{-\lambda_{\mathrm{eff}}}
$$

**Power-law decay**

$$
\beta_{\mathrm{eff}} = \frac{\beta_{type}}{(1 + \alpha \cdot impact)(1 + \rho \cdot stability)}
$$

$$
A_{t+1} = \frac{A_t}{(1 + \beta_{\mathrm{eff}})}
$$

여기서 $\alpha$는 impact modifier, $\rho$는 stability weight이다.

### 2.3 reinforcement 식

stability는 시간이 지나면 천천히 감소한다.

$$
S_{t+1} = \max(0, S_t (1 - \mu))
$$

direct recall 시에는 다음의 포화형 증가를 적용한다.

$$
S \leftarrow \min(S_{max}, S + r_{direct}(1 - S / S_{max}))
$$

cascade recall 시에는 더 약한 강화만 허용한다.

$$
S \leftarrow \min(S_{max}, S + r_{assoc} \cdot w_{assoc}(1 - S / S_{max}))
$$

이 설계는 reinforcement가 무한정 누적되는 것을 막고, direct recall이 association spreading보다 더 강한 학습 효과를 갖도록 강제한다.

### 2.4 reactivation 정책

실험은 세 가지 reactivation policy를 지원한다.

- `none`: 순수 감쇠 기준선
- `random`: 통제용 랜덤 재활성화
- `scheduled_query`: 고정 주기의 deterministic query-based reactivation

reinforcement 분석의 주 실험은 `scheduled_query`를 사용하고, baseline decay 비교는 `none`을 사용한다.

---

## 3. 평가 체계 재설계

### 3.1 Threshold Sweep

기존의 단일 threshold(0.3) 의존을 줄이기 위해 다음 grid를 기본으로 사용한다.

$$
T = \{0.2, 0.3, 0.4, 0.5\}
$$

각 threshold에서 recall/precision을 계산하고 평균을 취한다.

### 3.2 Primary / Secondary Scores

**Retrieval score**

$$
retrieval\_score = 0.7 \cdot recall\_mean + 0.3 \cdot precision\_mean
$$

**Plausibility score**

$$
plausibility\_score = 0.6 \cdot corr\_score + 0.4 \cdot smoothness\_score
$$

여기서 `corr_score`는 activation-recall correlation의 양의 방향만 보상하고, `smoothness_score`는 forgetting curve의 과도한 요동을 패널티화한다.

**Overall score**

$$
overall\_score = 0.7 \cdot retrieval\_score + 0.3 \cdot plausibility\_score
$$

이제 “최종 승자”는 `retrieval_score` 기준으로 먼저 해석하고, `overall_score`는 보조 요약값으로만 사용한다. `fact_episode_delta`는 최적화 대상에서 제외하고 진단용 metric으로만 유지한다.

---

## 4. 실험 구성

### 4.1 Canonical Dataset

이번 단계의 canonical dataset은 `data/memories_500.jsonl`이다. 모든 comparison, auto-improvement, figure 생성 스크립트는 이 데이터셋을 기준으로 맞춘다.

### 4.2 필수 실험 묶음

리포트는 다음 네 묶음으로 결과를 제시한다.

1. **무강화 decay 비교**  
   Exponential vs power law under `reactivation_policy="none"`

2. **reinforcement ablation**  
   no reinforcement / direct-heavy reinforcement / scheduled reinforcement 비교

3. **threshold 민감도 분석**  
   threshold grid에 따른 recall/precision 변화 및 retrieval winner의 안정성

4. **auto-improvement guidance 분석**  
   minimal / default / expert guidance의 `overall_score`, `retrieval_score`, `plausibility_score` 수렴 양상 비교

### 4.3 Auto-Improver 파라미터

AutoImprover는 다음 파라미터를 제안할 수 있다.

- `lambda_fact`, `lambda_episode`
- `beta_fact`, `beta_episode`
- `alpha`
- `stability_weight`
- `stability_decay`
- `reinforcement_gain_direct`
- `reinforcement_gain_assoc`
- `stability_cap`

iteration budget 기본값은 12, early-stop patience는 4이다.

---

## 5. 현재 해석 원칙

이 드래프트는 다음 해석 원칙을 따른다.

1. **단일 종합 점수만으로 모델 우열을 결론내리지 않는다.**
2. **retrieval 성능이 주 목적이고, plausibility는 보조 목적이다.**
3. **synthetic-data 기반 결과로 실제 인간 기억을 직접 주장하지 않는다.**
4. **reinforcement는 node stability 수준까지만 다루며, association edge learning은 다음 단계로 미룬다.**

이 원칙은 기존 리뷰에서 제기된 “지표와 결론의 불일치”, “단일 threshold의 임의성”, “synthetic data의 생태학적 타당성 부족” 문제에 대한 직접적인 대응이다.

---

## 6. 한계점

1. **합성 데이터 의존성**  
   여전히 실제 사용자 로그가 아니라 LLM 생성 한국어 메모리에 기반한다.

2. **실제 실험 결과 재산출 필요**  
   새 reinforcement-aware evaluator 기준의 수치 결과와 그림은 스크립트 재실행이 필요하다.

3. **동적 edge learning 미포함**  
   이번 단계에서는 node stability만 강화하며, association weight 자체의 장기 강화/약화는 구현하지 않았다.

4. **언어 특화성**  
   현재는 한국어 임베딩 및 한국어 synthetic memory를 기준으로 설계되어 있다.

5. **LLM 최적화의 탐색 예산 한계**  
   budget을 12로 늘렸지만 전역 최적을 보장하지는 않는다.

---

## 7. 재현 경로

개발/검증:

```bash
uv sync --extra dev
PYTHONPATH=src uv run pytest -q
```

기본 시뮬레이션:

```bash
PYTHONPATH=src uv run python -m memory_decay.main \
  --dataset data/memories_500.jsonl \
  --decay-type exponential \
  --reactivation-policy scheduled_query \
  --total-ticks 200 \
  --eval-interval 20
```

auto-improvement:

```bash
PYTHONPATH=src uv run python scripts/run_auto_improve.py
```

시각화:

```bash
PYTHONPATH=src uv run python scripts/visualize.py data/comparison_results_korean_emb.json docs/figures
```

---

## 8. 결론

이 재설계의 핵심은 기억 모델을 `잊힘만 있는 activation decay`에서 `activation + stability reinforcement` 구조로 바꾸고, 평가를 `single-threshold composite score`에서 `threshold-robust retrieval + plausibility` 체계로 재편한 데 있다. 이로써 기존 리뷰에서 드러난 핵심 문제였던 평가 목적의 불명확성, threshold 민감도, reinforcement 부재, 과도한 human-memory claim을 코드와 문서 차원에서 동시에 줄였다.  

다음 실제 연구 단계는 새 evaluator 기준으로 결과를 재산출하고, 합성 데이터가 아닌 실제 사용자 상호작용 로그에서 같은 경향이 유지되는지를 검증하는 것이다.
