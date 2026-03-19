# 감쇠 함수 수학적 분석 메모

**날짜**: 2026-03-18
**분석자**: 동역학계 수학자 리뷰 (이전 인지과학 분석 대체)
**현재 best**: exp_0298, overall = 0.2214
**기준선**: exp_0000, overall = 0.0281

---

## 1. 현재 상태 정밀 진단

### 1.1 실험 이력 요약

- 296개의 이전 실험은 구 채점/데이터 기반으로 **무효화됨**
- 새 채점 체계에서 기준선(exp_0000): overall = 0.0281
- 현재 최고(exp_0298): overall = 0.2214 (시그모이드 게이트 바닥 + 불연속 가지치기)
- 구 체계의 exp_0293 (reciprocal decay): recall_mean = 0.372, overall_old = 0.274

### 1.2 이론적 상한

`similarity_recall_rate = 0.390`은 임베딩 검색의 임계값 무관 상한이다.

```
recall_mean 상한 = 0.390
MRR 상한 ≈ 0.390 (현실적 추정)
precision_lift 상한 ≈ 0.10

retrieval_max = 0.55 * 0.39 + 0.30 * 0.39 + 0.15 * 0.10 = 0.3465
overall_max = 0.3465 * (0.85 + 0.15 * 1.0) = 0.3465

현재 best 대비 개선 여지: (0.3465 - 0.2214) / 0.2214 ≈ 57%
```

### 1.3 병목의 정량적 분해

exp_0298의 retrieval 성분별 기여:

| 성분 | 현재값 | 상한 | 가중치 | 기여 (현재) | 기여 (상한) | 갭 |
|------|--------|------|--------|-------------|-------------|-----|
| recall_mean | 0.302 | 0.390 | 0.55 | 0.166 | 0.215 | 0.049 |
| mrr_mean | 0.220 | 0.390 | 0.30 | 0.066 | 0.117 | 0.051 |
| precision_lift | 0.015 | 0.100 | 0.15 | 0.002 | 0.015 | 0.013 |
| **합계** | | | | **0.234** | **0.347** | **0.113** |

**recall과 MRR의 갭이 거의 같은 크기(~0.05)**이다. precision_lift는 절대값이 작다.
plausibility 승수: 현재 0.633 → 최대 1.0, 효과는 retrieval * 0.15 * Δplaus ≈ 0.01 수준.

**결론: retrieval_score, 특히 recall과 MRR 개선이 지배적 과제다.**

---

## 2. 현재 함수의 동역학적 분석

### 2.1 exp_0298의 구조

```
importance = (α·i + ρ·s) / (α + ρ)
floor = f_max · σ(k·(importance - m))
λ_eff = λ_τ / (1 + α·i + ρ·s) · P(importance)
a_{t+1} = floor + (a_t - floor) · exp(-λ_eff)
```

여기서 P(w) = prune_factor if w < sigmoid_mid else 1 (불연속).

### 2.2 고정점 분석

순수 감쇠(재활성화 없음) 하에서 이 사상의 고정점은 `a* = floor`이다.

고정점에서의 야코비안(1차원이므로 미분):
```
∂a_{t+1}/∂a_t = exp(-λ_eff)
```

이것은 항상 `(0, 1)` 구간에 있으므로, floor는 **안정 고정점**이다. 수렴 속도는 `exp(-λ_eff)`에 의해 결정된다.

### 2.3 문제: 모든 기억이 동일한 궤적 형태를 가짐

`a_t - floor = (a_0 - floor) · exp(-λ_eff · t)` 형태이므로, **모든 기억이 단순히 초기값과 바닥이 다른 동일 형태의 지수 감쇠**를 따른다. 이것은:

1. 임계값 [0.2, 0.3, 0.4, 0.5]을 통과하는 시점이 floor에 의해 거의 결정됨
2. threshold_discrimination = 0.171로 제한적
3. **activation과 recall의 상관이 낮음** (corr = 0.001) — floor가 높은 기억이 반드시 검색 가능한 기억과 일치하지 않음

---

## 3. 대안 함수형의 수학적 분류

감쇠 함수를 일반적으로 다음과 같이 분류한다:

### 3.1 분류 체계

excess를 `e_t = a_t - floor`로 정의하면, 감쇠 사상은 excess에 대한 사상으로 환원된다:

| 유형 | 사상 | 수렴 속도 | 특성 |
|------|------|-----------|------|
| 지수 | e_{t+1} = e_t · exp(-λ) | 기하급수 | 무기억, 비율 일정 |
| 쌍곡선 | e_{t+1} = e_t / (1 + r) | 조화급수 ∼ 1/t | 두꺼운 꼬리, 느린 수렴 |
| 멱법칙 | e_{t+1} = e_t · (t/(t+1))^β | t^{-β} | 시간 의존, 점점 느려짐 |
| 늘린 지수 | e_{t+1} = e_t · exp(-(λ)^β) | 가변적 | β로 곡률 제어 |
| 이중 지수 | e_{t+1} = w·e_t·e^{-λ_1} + (1-w)·e_t·e^{-λ_2} | 초기: 혼합, 후기: 느린 성분 지배 | 두 시상수 분리 |

### 3.2 핵심 수학적 통찰

**정리**: 단조감소 제약 하에서, excess의 수렴 속도가 `O(e^{-λt})`보다 느린 함수형(쌍곡선, 멱법칙)은 임의의 시점에서 더 높은 activation을 유지한다.

**증명 스케치**: `e_t^{(exp)} = C·e^{-λt}`와 `e_t^{(hyp)} = C/(1+rt)`를 비교하면, 충분히 큰 t에 대해 항상 `e_t^{(hyp)} > e_t^{(exp)}`이다.

**함의**: recall_mean을 높이려면 (즉, 더 많은 기억이 임계값 위에 남으려면), **지수보다 느린 수렴 형태가 유리하다**. 이것은 exp_0293의 reciprocal decay가 높은 recall을 보인 이유를 정확히 설명한다.

### 3.3 MRR을 높이기 위한 조건

MRR은 정답이 top-k 중 높은 순위에 있어야 높다. 이를 위해서는 **정답 기억의 activation이 비정답보다 높아야** 한다. 즉:

```
Corr(activation, is_correct_answer) > 0 이 필요
```

이것은 impact와 "검색 가능성"의 정렬이 필요하다. 현재 시스템에서 impact는 데이터에 주어진 값이고, 검색 가능성은 임베딩 유사도에 의해 결정된다. 이 둘은 독립적이므로 **correlation을 높이려면 감쇠 함수가 추가 정보를 활용해야 한다**.

그러나 compute_decay의 인터페이스는 `(activation, impact, stability, mtype, params)`로 고정되어 있으므로, 유일하게 활용 가능한 추가 정보는 **stability의 시간 변화 패턴**이다.

---

## 4. 구체적 함수 제안

### 4.1 제안 A: 쌍곡선 감쇠 + 시그모이드 바닥 + 연속 가지치기

**근거**: 쌍곡선 감쇠(reciprocal)가 이미 높은 recall을 보였다. 새 채점 체계에서의 불연속 가지치기를 연속화하고, 시그모이드 바닥을 유지한다.

```python
import math

def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # 시그모이드 바닥 (impact만 사용, stability는 rate에만)
    k = params.get("sigmoid_k", 15.0)
    mid = params.get("sigmoid_mid", 0.25)
    floor_max = params.get("floor_max", 0.45)

    if impact - mid >= 0:
        sig = 1.0 / (1.0 + math.exp(-k * (impact - mid)))
    else:
        z = math.exp(k * (impact - mid))
        sig = z / (1.0 + z)

    floor = min(floor_max * sig, activation)

    # 쌍곡선 감쇠율
    base_rate = params.get("base_rate", 0.015)
    type_factor = params.get("fact_factor", 0.7) if mtype == "fact" else 1.0
    stab_factor = 1.0 + params.get("stability_weight", 1.0) * stability

    rate = base_rate * type_factor / stab_factor

    # 연속적 가지치기: sig → 0일수록 rate 증가
    prune_boost = params.get("prune_boost", 3.0)
    rate *= 1.0 + prune_boost * (1.0 - sig)

    excess = max(activation - floor, 0.0)
    new_excess = excess / (1.0 + rate)
    return min(floor + new_excess, activation)
```

**수학적 성질**:
- excess 수렴: O(1/t) — 지수보다 느림, recall 유지에 유리
- 바닥과 가지치기가 모두 impact의 연속 함수 — smoothness 보장
- stability가 rate에만 영향 → 재활성화 효과가 감쇠 속도에 직접 작용

### 4.2 제안 B: 이중 시상수 지수 감쇠 (Bi-exponential)

**근거**: correlation/MRR 직접 겨냥. 중요한 기억은 느린 성분이 지배, 덜 중요한 기억은 빠른 성분이 지배하면 activation 분포가 impact와 정렬된다.

```python
import math

def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # 시그모이드 바닥
    k = params.get("sigmoid_k", 15.0)
    mid = params.get("sigmoid_mid", 0.25)
    floor_max = params.get("floor_max", 0.45)

    if impact - mid >= 0:
        sig = 1.0 / (1.0 + math.exp(-k * (impact - mid)))
    else:
        z = math.exp(k * (impact - mid))
        sig = z / (1.0 + z)

    floor = min(floor_max * sig, activation)

    # 두 시상수
    lam_slow = params.get("lam_slow", 0.005)
    lam_fast = params.get("lam_fast", 0.06)
    stab_factor = 1.0 + params.get("stability_weight", 1.0) * stability

    # 중요도에 따라 성분 혼합
    w_slow = sig  # 중요할수록 느린 성분 비중 증가

    excess = max(activation - floor, 0.0)
    slow = w_slow * excess * math.exp(-lam_slow / stab_factor)
    fast = (1.0 - w_slow) * excess * math.exp(-lam_fast / stab_factor)

    # 유형 보정: episode는 fast 성분 가속
    if mtype == "episode":
        fast *= math.exp(-params.get("episode_extra", 0.01))

    return min(floor + slow + fast, activation)
```

**수학적 성질**:
- 초기: fast 성분이 빠르게 감쇠 → 낮은 중요도 기억이 먼저 떨어짐
- 후기: slow 성분만 남음 → 높은 중요도 기억이 임계값 위에 오래 잔류
- impact-activation 상관관계가 시간에 따라 자연스럽게 형성됨

### 4.3 제안 C: 활성화 의존 감쇠율 (Jost 법칙의 엄밀한 이산화)

**근거**: Jost(1897)의 법칙 — "같은 강도의 두 기억 중 오래된 것이 더 느리게 잊힌다" — 을 수학적으로 엄밀하게 이산화한다. activation이 낮을수록(= 오래됨) 감쇠율이 낮아지는 **양의 피드백**을 가진 비선형 사상.

```python
import math

def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # 시그모이드 바닥
    k = params.get("sigmoid_k", 15.0)
    mid = params.get("sigmoid_mid", 0.25)
    floor_max = params.get("floor_max", 0.45)

    if impact - mid >= 0:
        sig = 1.0 / (1.0 + math.exp(-k * (impact - mid)))
    else:
        z = math.exp(k * (impact - mid))
        sig = z / (1.0 + z)

    floor = min(floor_max * sig, activation)

    # Jost 감쇠: rate ∝ activation^gamma
    base_rate = params.get("base_rate", 0.025)
    gamma = params.get("gamma", 0.5)  # 0 < gamma < 1: sublinear
    type_factor = params.get("fact_factor", 0.7) if mtype == "fact" else 1.0
    stab_factor = 1.0 + params.get("stability_weight", 1.0) * stability

    # 정규화된 excess
    excess = max(activation - floor, 0.0)
    total_range = max(1.0 - floor, 1e-9)
    norm_excess = excess / total_range

    # 감쇠율이 바닥으로부터의 거리에 의존
    effective_rate = base_rate * type_factor * (norm_excess ** gamma) / stab_factor

    # 가지치기 (연속)
    prune_boost = params.get("prune_boost", 2.0)
    effective_rate *= 1.0 + prune_boost * (1.0 - sig)

    return min(floor + excess * math.exp(-effective_rate), activation)
```

**수학적 성질**:
- gamma < 1일 때: norm_excess가 작을수록 rate가 0에 접근 → **바닥 근처에서 감쇠가 극적으로 느려짐**
- 임계값 구간을 통과하는 시간이 길어짐 → threshold_discrimination 증가
- 이것은 stretched exponential의 이산 반복 버전과 동치

### 4.4 제안 D: 멱법칙 (stability를 시간 proxy로)

```python
import math

def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # 시그모이드 바닥
    k = params.get("sigmoid_k", 15.0)
    mid = params.get("sigmoid_mid", 0.25)
    floor_max = params.get("floor_max", 0.45)

    if impact - mid >= 0:
        sig = 1.0 / (1.0 + math.exp(-k * (impact - mid)))
    else:
        z = math.exp(k * (impact - mid))
        sig = z / (1.0 + z)

    floor = min(floor_max * sig, activation)

    # stability에서 유효 나이 역산
    # stability_decay = 0.01이므로 s_t = s_0 * 0.99^t
    # t = -ln(s/s_0) / ln(0.99) ≈ 100 * ln(s_0/s)
    # s_0 = 초기 stability ≈ 0 (reinforcement 전) 또는 gain 후 값
    # s가 0에 가까우면 매우 오래된 기억
    s_decay = params.get("stability_decay", 0.01)
    eff_age = max(-math.log(max(stability, 1e-6)) / max(-math.log(1.0 - s_decay), 1e-9), 1.0)
    eff_age = min(eff_age, 1000.0)  # 안전 상한

    beta = params.get("beta", 0.4)
    type_factor = params.get("fact_factor", 0.8) if mtype == "fact" else 1.0
    impact_mod = 1.0 + params.get("alpha", 1.5) * impact

    # 멱법칙: (t/(t+1))^(beta/impact_mod)
    decay_factor = (eff_age / (eff_age + 1.0)) ** (beta * type_factor / impact_mod)

    excess = max(activation - floor, 0.0)
    return min(floor + excess * decay_factor, activation)
```

**주의**: stability가 0에서 시작하고 reinforcement 후에만 양수가 되므로, 초기 stability = 0인 기억은 eff_age → ∞ → decay_factor → 1 (거의 감쇠 없음). 이것은 **원하는 동작의 반대**이다.

**수정**: stability = 0인 경우를 별도 처리하거나, 초기값을 가정해야 한다. 이 때문에 이 제안의 위험도가 높다.

---

## 5. 실험 우선순위

| 순위 | 제안 | 핵심 메커니즘 | 기대 개선 | 위험도 |
|------|------|---------------|-----------|--------|
| **1** | A: 쌍곡선 + 연속 게이트 | O(1/t) 수렴, 연속 가지치기 | recall +0.05, smoothness 유지 | **낮음** |
| **2** | B: 이중 시상수 | fast/slow 분리, 자연 상관 | corr +0.1, MRR +0.03 | 중간 |
| **3** | C: Jost 감쇠 | activation 의존 rate | discrimination 개선 | 중간 |
| 4 | D: 멱법칙 | stability 시간 proxy | 전체적 | **높음** |

---

## 6. 제안 A와 B의 결합: 쌍곡선-이중속도 하이브리드

가장 야심적인 제안으로, A와 B의 장점을 결합한다:

```python
import math

def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # 시그모이드 바닥
    k = params.get("sigmoid_k", 15.0)
    mid = params.get("sigmoid_mid", 0.25)
    floor_max = params.get("floor_max", 0.45)

    if impact - mid >= 0:
        sig = 1.0 / (1.0 + math.exp(-k * (impact - mid)))
    else:
        z = math.exp(k * (impact - mid))
        sig = z / (1.0 + z)

    floor = min(floor_max * sig, activation)
    excess = max(activation - floor, 0.0)
    if excess <= 0:
        return activation

    stab_factor = 1.0 + params.get("stability_weight", 1.0) * stability

    # 느린 성분: 쌍곡선 (높은 중요도 기억용)
    r_slow = params.get("r_slow", 0.008) / stab_factor
    slow_excess = excess / (1.0 + r_slow)

    # 빠른 성분: 지수 (낮은 중요도 기억 가지치기용)
    lam_fast = params.get("lam_fast", 0.05) / stab_factor
    fast_excess = excess * math.exp(-lam_fast)

    # 유형 보정
    if mtype == "episode":
        lam_fast *= params.get("episode_accel", 1.3)
        fast_excess = excess * math.exp(-lam_fast / stab_factor)

    # 중요도에 따른 혼합
    w_slow = sig
    new_excess = w_slow * slow_excess + (1.0 - w_slow) * fast_excess

    return min(floor + new_excess, activation)
```

**성질**: 높은 impact 기억은 쌍곡선(두꺼운 꼬리)을 따라 오래 유지, 낮은 impact 기억은 지수(빠른 감쇠)를 따라 빨리 제거. 이 분리가 correlation과 precision_lift를 동시에 개선할 수 있다.

---

## 7. 결론

### 핵심 수학적 통찰

1. **지수 감쇠의 무기억 성질이 근본 한계**이다. 모든 기억이 동일한 형태의 궤적을 따르므로, activation 분포가 impact와 정렬되지 않는다.

2. **쌍곡선 감쇠의 O(1/t) 수렴이 recall 유지에 본질적으로 유리**하다. 이것은 exp_0293의 성공이 우연이 아님을 수학적으로 확인한다.

3. **두 감쇠 속도의 혼합(bi-rate mixing)이 correlation 개선의 열쇠**이다. 중요한 기억과 덜 중요한 기억의 activation 궤적이 발산해야 한다.

4. **불연속 가지치기를 연속화**하면 smoothness가 개선되고, 이는 plausibility 승수를 통해 overall에 기여한다.

### 실행 권장

제안 A (쌍곡선)를 먼저 실행하여 새 채점 체계에서의 reciprocal decay 성능을 확인한 후, 제안 B (이중 시상수)와 결합 제안을 순차적으로 시도할 것을 권장한다. 파라미터 수가 7-9개로 416개 데이터 대비 과적합 위험이 낮으므로, 자유도를 최대한 활용하는 것이 합리적이다.
