# Memory Decay Function 자동 탐색 리포트

## 개요

메모리 그래프 시스템의 **망각 함수(decay function)**를 자동으로 탐색하여 최적의 decay 형태와 파라미터를 발견하는 실험을 진행했다. 총 **25개 실험**을 3개 cycle에 걸쳐 수행했으며, baseline 대비 **+63.4%** 개선을 달성했다.

---

## 평가 체계

### Scoring Formula

```
overall = 0.7 × retrieval + 0.3 × plausibility

retrieval   = 0.7 × recall_mean + 0.3 × precision_mean
plausibility = 0.6 × correlation + 0.4 × smoothness
```

각 구성 요소의 overall에 대한 실질 가중치:

| 구성 요소       | 가중치 | 설명 |
|----------------|--------|------|
| **recall_mean**    | **0.49** | 4개 threshold (0.2, 0.3, 0.4, 0.5)에서의 recall 평균 |
| precision_mean | 0.21   | 동일 threshold에서의 precision 평균 |
| correlation    | 0.18   | activation과 recall 성공의 Pearson 상관 |
| smoothness     | 0.12   | 망각 곡선의 부드러움 (variance 기반) |

### 시뮬레이션 조건

- **200 tick** 시뮬레이션, 20 tick 간격 평가
- `scheduled_query` 재활성화 정책 (10 tick 간격, boost=0.3)
- Threshold sweep: [0.2, 0.3, 0.4, 0.5]
- Recall 이론적 상한: **~0.498** (embedding similarity search 품질에 의해 결정)

---

## 결과 요약

### 진화 경로

```
Baseline (exponential)     ████░░░░░░░░░░░░░░░░  0.242
exp_0001 (quadratic)       █████████░░░░░░░░░░░  0.344  (+42.0%)
exp_0003 (+ exp impact)    ██████████░░░░░░░░░░  0.364  (+50.2%)
exp_0004 (cubic)           ██████████████░░░░░░  0.393  (+62.3%)
exp_0013 (ratio=3.0)       ██████████████░░░░░░  0.395  (+63.1%)
exp_0019 (λ scale 0.8x)    ██████████████░░░░░░  0.396  (+63.4%)  ← BEST
```

### 최종 최적 해: exp_0019

```python
def compute_decay(activation, impact, stability, mtype, params):
    combined = exp(2.0 × impact) × (1 + 0.8 × stability)
    λ = 0.008 (fact) / 0.024 (episode)
    decay_rate = λ × activation² / combined
    return activation × (1 - decay_rate)
```

| 메트릭 | Baseline | Best (exp_0019) | 변화 |
|--------|----------|-----------------|------|
| **overall** | **0.242** | **0.396** | **+63.4%** |
| retrieval | 0.041 | 0.369 | +799% |
| recall_mean | 0.019 | 0.484 | +2447% |
| precision | 0.093 | 0.099 | +6% |
| correlation | 0.531 | 0.100 | -81% |
| smoothness | 0.983 | 1.000 | +2% |
| plausibility | 0.712 | 0.460 | -35% |

---

## Cycle별 상세 분석

### Cycle 1: Decay Shape 탐색 (exp_0000 ~ exp_0008)

**핵심 발견: Power-law tail이 exponential을 압도한다.**

Baseline의 exponential decay `a × exp(-λ/combined)`는 200 tick 후 activation을 ~0.009까지 떨어뜨려 recall이 2%에 불과했다. 수학적 형태를 변경하여 power-law tail을 도입한 것이 가장 큰 단일 개선이었다.

| Decay 형태 | 수식 | 점근 꼬리 | tick 200 (impact=0) | overall |
|-----------|------|----------|---------------------|---------|
| Exponential | `a × exp(-λ)` | `e^{-t}` | 0.009 | 0.242 |
| Quadratic | `a × (1 - λa/c)` | `1/t` | 0.227 | 0.344 |
| **Cubic** | **`a × (1 - λa²/c)`** | **`1/√t`** | **0.354** | **0.393** |
| Quartic | `a × (1 - λa³/c)` | `1/t^{1/3}` | 0.415 | 0.392 |

**Cubic (a³ rate)이 최적인 이유:**
- Quadratic (1/t tail): recall_mean=0.303 — 너무 많은 항목이 threshold 아래로 떨어짐
- Cubic (1/√t tail): recall_mean=0.494 — 거의 모든 항목이 threshold 위에 유지
- Quartic (1/t^{1/3} tail): recall_mean=0.498 — 최대지만 correlation이 0.04로 급감

Cubic은 recall을 거의 최대화하면서 약간의 correlation 신호를 유지하는 sweet spot이다.

**Impact 보호 방식의 진화:**

```
v1 (baseline):  combined = (1 + 0.5 × impact) × (1 + 0.8 × stability)
v2 (exp_0003):  combined = exp(2.0 × impact) × (1 + 0.8 × stability)
```

선형 `(1 + α × impact)` → 지수 `exp(α × impact)` 변환으로 high-impact 항목의 보호가 비선형적으로 강화됨.

**시도했으나 실패한 접근:**

| 접근 | 실험 | 결과 | 실패 원인 |
|-----|------|------|----------|
| Impact floor | exp_0002 | 0.344 (no gain) | Floor이 power-law tail보다 항상 낮아 활성화되지 않음 |
| Equilibrium 수렴 | exp_0006 | 0.375 (worse) | exp(α×impact)로 인해 수렴이 너무 느림 |
| 강화된 reinforcement | exp_0007 | 0.394 (no gain) | 200 tick 내 재활성화가 0~2회로 너무 적음 |
| Memory consolidation | exp_0009 | 0.385 (worse) | 성장하는 항목이 모든 threshold 통과 → correlation = 0 |

### Cycle 2: Episode:Fact 비율 최적화 (exp_0009 ~ exp_0015)

**핵심 발견: Recall-correlation trade-off의 최적점이 ratio=3.0에 존재한다.**

Episode와 fact에 동일한 lambda를 사용하면 (ratio=1.0) recall은 최대(0.498)지만 correlation이 0이 된다. Ratio를 높이면 episode가 더 빨리 decay하여 fact/episode 구분이 생기고, correlation이 증가한다. 하지만 recall이 감소한다.

```
ratio=1.0  ██████████████████████████████  recall=0.498  corr=0.000  → 0.385
ratio=1.5  █████████████████████████████░  recall=0.497  corr=0.030  → 0.390
ratio=2.0  ████████████████████████████░░  recall=0.494  corr=0.058  → 0.393
ratio=2.5  ██████████████████████████░░░░  recall=0.481  corr=0.100  → 0.395
ratio=3.0  █████████████████████████░░░░░  recall=0.472  corr=0.126  → 0.395 ← PEAK
ratio=4.0  ███████████████████████░░░░░░░  recall=0.451  corr=0.190  → 0.394
ratio=5.0  ██████████████████████░░░░░░░░  recall=0.440  corr=0.228  → 0.393
```

**최적 비율의 수학적 해석:**

Pareto frontier 위에서 `dR/dC = -(0.494-0.408)/(0.293-0.058) ≈ -0.366`이며, scoring formula의 최적 기울기 `-0.18/0.49 ≈ -0.367`과 거의 일치한다. Quadratic fit으로 정확한 peak 위치를 계산하면 **ratio ≈ 3.04**.

### Cycle 3: 파라미터 미세 조정 (exp_0016 ~ exp_0024)

**핵심 발견: ratio=3.0 고정 시, 절대 lambda 스케일의 최적점이 0.8x에 존재한다.**

세 가지 축을 탐색:

**1. Alpha (impact 보호 강도)**

| alpha | overall | 분석 |
|-------|---------|------|
| 1.5   | 0.393   | 보호 약화 → 항목 사망 증가 |
| **2.0** | **0.395** | **최적** |
| 2.5   | 0.395   | 동일 (이미 충분한 보호) |

**2. 절대 Lambda 스케일 (ratio=3.0 고정)**

| λ_fact / λ_episode | overall | recall_mean | corr |
|---------------------|---------|-------------|------|
| 0.004 / 0.012 | 0.3956 | 0.498 | 0.061 |
| 0.006 / 0.018 | 0.3958 | 0.496 | 0.067 |
| 0.007 / 0.021 | 0.3960 | 0.491 | 0.081 |
| **0.008 / 0.024** | **0.3960** | **0.484** | **0.100** |
| 0.009 / 0.027 | 0.3956 | 0.478 | 0.113 |
| 0.010 / 0.030 | 0.3951 | 0.472 | 0.126 |
| 0.012 / 0.036 | 0.3929 | 0.458 | 0.164 |

Peak은 0.007~0.008에서 broad하게 형성 (차이 < 0.0001).

**3. 혼합 Decay Shape**

Cubic facts + quadratic episodes (ratio=3.0): overall=0.3945. 순수 cubic 대비 약간 열세.

---

## 구조적 한계 분석

### Recall 상한: 0.498

모든 항목이 threshold 위에 있어도 recall은 ~0.498에서 상한이 걸린다. 이는 embedding similarity search에서 test query의 target이 top-5 결과에 포함되는 비율이 49.8%이기 때문이다. Decay function으로는 이 상한을 높일 수 없다.

### Precision 고정: ~0.10

25개 실험 전체에서 precision은 0.093~0.101 범위에서 거의 변하지 않았다. Top-5 similarity 결과 중 관련 항목의 비율은 그래프 연관 구조와 embedding 품질에 의해 결정되며, decay function의 영향을 받지 않는다.

### Recall-Correlation Trade-off

```
            correlation
       0.5 ┤ ●baseline
           │
       0.3 ┤         ●exp_0005
           │              ●exp_0014
       0.2 ┤
           │        ●exp_0019(best)
       0.1 ┤   ●exp_0004
           │ ●exp_0008
       0.0 ┤●exp_0010
           └──────────────────────
            0.0  0.2  0.4  recall  0.5
```

위 그래프에서 보이듯이, recall과 correlation은 근본적으로 trade-off 관계이다. 모든 항목이 threshold 위에 있으면 (recall → max) correlation 신호가 사라진다. Scoring formula에서 recall의 가중치(0.49)가 correlation(0.18)의 **2.7배**이므로, recall을 최대화하는 방향이 항상 유리하다.

### 이론적 상한 추정

```
overall_max = 0.49 × 0.498 + 0.21 × 0.10 + 0.18 × corr_opt + 0.12 × 1.0
            = 0.244 + 0.021 + 0.120 + 0.18 × corr_opt
            = 0.385 + 0.18 × corr_opt
```

현재 best의 corr=0.100 → overall=0.385 + 0.018 = **0.396** (실측치와 일치).
Recall 최대에서 달성 가능한 최대 correlation은 ~0.10이므로 (모든 항목이 threshold 위에 있을 때), **현재 0.396이 사실상 이론적 상한에 근접**한다.

---

## 최종 최적 파라미터

```python
# Decay function
def compute_decay(activation, impact, stability, mtype, params):
    combined = exp(2.0 × impact) × (1 + 0.8 × stability)
    λ = 0.008 if mtype == "fact" else 0.024
    return activation × (1 - λ × activation² / combined)

# Simulation parameters
{
    "lambda_fact": 0.008,
    "lambda_episode": 0.024,
    "alpha": 2.0,              # exp(alpha × impact) 보호
    "stability_weight": 0.8,    # stability → combined factor
    "stability_decay": 0.01,    # stability 감쇠율
    "reinforcement_gain_direct": 0.2,
    "reinforcement_gain_assoc": 0.05,
    "stability_cap": 1.0
}
```

### Tick별 성능 프로필 (Best: exp_0019)

| Tick | Recall@0.3 | Recall@0.5 | Overall |
|------|-----------|-----------|---------|
| 0    | 0.498     | 0.498     | 0.325   |
| 40   | 0.498     | 0.498     | 0.399   |
| 80   | 0.498     | 0.498     | 0.407   |
| 120  | 0.498     | 0.498     | 0.401   |
| 160  | 0.498     | 0.488     | 0.400   |
| 200  | 0.498     | 0.442     | 0.396   |

Tick 160까지 recall이 거의 최대치를 유지하며, threshold 0.5에서만 tick 160 이후 완만하게 감소한다.

---

## 핵심 교훈

1. **수학적 형태 > 파라미터 튜닝**: Exponential → power-law 전환이 전체 개선의 95%를 차지 (+0.151 / +0.154)
2. **Recall이 왕**: Scoring formula에서 recall 1%의 가치(0.0049)가 correlation 1%의 가치(0.0018)의 2.7배
3. **Cubic이 sweet spot**: Quadratic은 너무 빠른 decay, quartic은 너무 느린 decay. Cubic의 1/√t tail이 recall과 correlation의 최적 균형
4. **Episode:fact 비율은 correlation의 유일한 원천**: 동일 lambda → corr=0, 비율=3.0에서 최적
5. **Precision은 decay function으로 개선 불가**: Embedding 품질과 그래프 구조에 의해 결정

---

## 향후 연구 방향

현재 0.396에서 추가 개선을 위한 가능한 경로:

| 방향 | 예상 효과 | 난이도 |
|-----|----------|--------|
| Embedding 모델 개선 | Recall 상한 0.498 → 0.6+ 가능 | 높음 |
| 데이터셋 품질 개선 | Precision 0.10 → 0.15+ 가능 | 중간 |
| Scoring formula 재설계 | Correlation 가중치 증가 시 다른 최적해 존재 | 낮음 |
| 비정상(non-stationary) decay | 시간에 따라 변하는 decay rate | 중간 |
| 연관 구조 기반 decay | Impact 대신 graph centrality 활용 | 높음 |

---

*25 experiments, 3 cycles, ~25 minutes compute time.*
*Generated: 2026-03-18*
