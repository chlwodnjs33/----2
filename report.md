# PAANE: Physics-Anchored Adaptive NLOS Ensemble을 이용한 5G 실내 측위

## 1. 모티베이션 & 인트로

### 데이터 분석 및 문제 정의

본 과제는 InF-DH(Indoor Factory, Dense Hotspot) FR1 환경에서 18개 기지국의 RTT(Round-Trip Time) 측정값만을 이용하여 사용자 위치를 추정하는 문제이다. 제공된 훈련 데이터(700명)를 분석한 결과, 이 환경의 핵심 특성은 심각한 NLOS(Non-Line-of-Sight) 오염이었다.

| 분석 항목 | 수치 |
|---|---|
| NLOS 측정 비율 (d_hat > 실제 거리) | 81.4% |
| 평균 NLOS bias | +15.93 m |
| 중앙값 NLOS bias | +10.14 m |
| 18개 BS 중 절반 이상이 NLOS인 사용자 비율 | 61.1% |
| 실제 거리와 NLOS bias 간 Spearman 상관계수 | 0.17 |

특히 마지막 항목이 핵심 인사이트였다. NLOS bias와 실제 거리의 상관이 0.17에 불과하다는 것은, 가까운 기지국이라도 NLOS일 수 있고, 먼 기지국이 LOS일 수도 있음을 의미한다. 즉, 단순히 가까운 기지국을 선택하거나 거리 기반 가중치를 부여하는 방식으로는 NLOS를 효과적으로 회피할 수 없다.

### 중간 단계 실험과 알고리즘 도출 과정

초기에는 여러 물리 기반 삼변측량 방법을 비교 실험하였다.

| 방법 | 평균 오차 |
|---|---|
| LS 삼변측량 (일반 최소제곱, 18 BS) | 23.20 m |
| 역거리 가중 중심 (IDW centroid) | 23.34 m |
| WLS 삼변측량 | 33.53 m |
| Cauchy-robust NLS (18 BS 전체) | 11.10 m |
| Cauchy-robust NLS (가장 가까운 6 BS) | 9.63 m |

Cauchy loss 기반 비선형 최소제곱이 가장 우수하였으나, 물리 추정 단독으로는 9~11 m 수준의 한계가 있었다. 이는 NLOS bias가 물리적 모델로 완전히 제거되지 않기 때문이다.

여기서 두 가지 아이디어가 도출되었다. 첫째, 물리 추정(anchor)의 잔차(residual)를 기계학습으로 추가 보정한다. 둘째, 어떤 기지국의 측정값이 신뢰할 만한가에 대한 정보를 피처로 표현하여 모델이 학습하도록 한다. 이 두 아이디어를 결합하여 PAANE(Physics-Anchored Adaptive NLOS Ensemble)을 설계하였다.

---

## 2. 알고리즘 설명

PAANE는 물리 추정, 피처 생성, 기계학습 앙상블의 3단계로 구성된다.

### 단계 1: Cauchy-robust NLS Anchor 추정

18개 기지국 위치 $\mathbf{b}_i \in \mathbb{R}^2$와 RTT 측정 거리 $\hat{d}_i$가 주어졌을 때, 사용자 위치 $\mathbf{p}$를 다음 목적함수를 최소화하여 추정한다.

$$\hat{\mathbf{p}}_\text{anchor} = \arg\min_{\mathbf{p}} \sum_{i=1}^{18} \rho_\text{Cauchy}\!\left(\frac{\|\mathbf{p} - \mathbf{b}_i\|_2 - \hat{d}_i}{c}\right)$$

여기서 $\rho_\text{Cauchy}(r) = \log(1 + r^2)$이고 스케일 파라미터 $c = 5.0$이다. Cauchy loss는 잔차가 클수록 영향을 자동으로 억제하는 M-estimator로, 극단적인 NLOS 측정값이 추정을 왜곡하는 것을 방지한다. 초기값은 역거리 가중 중심으로 설정하고 경계는 기지국 좌표 범위에 여유(margin)를 더한 값으로 제한한다.

### 단계 2: 다중 해상도 Sub-Anchor 계산

Main anchor 외에, RTT 측정값 기준으로 가장 가까운 $k$개 기지국만을 선택하여 동일한 Cauchy-robust NLS를 적용한 sub-anchor를 추가로 계산한다. $k \in \{4, 6, 8\}$의 세 가지 sub-anchor를 생성한다. 이는 서로 다른 크기의 BS 부분집합에서 얻은 독립적인 추정치로, 측정 환경의 다양한 NLOS 패턴을 커버한다.

### 단계 3: 187D 피처 벡터 생성

다음 5개 카테고리의 피처를 결합하여 총 187차원 입력 벡터를 구성한다.

**기존 피처 (101D):**

| 피처 | 차원 | 계산 방법 |
|---|---|---|
| Raw RTT | 18 | 측정값 $\hat{d}_i$ 그대로 |
| Anchor 위치 | 2 | 단계 1의 $\hat{\mathbf{p}}_\text{anchor}$ |
| Range residual | 18 | $r_i = \hat{d}_i - \|\hat{\mathbf{p}}_\text{anchor} - \mathbf{b}_i\|_2$ |
| Abs residual | 18 | $|r_i|$ |
| 통계 | 13 | mean, std, min, max, median, 최소 4개, 최대 4개 |
| WLS 삼변측량 | 2 | 가중 최소제곱 위치 |
| IDW 중심 | 2 | 역거리 가중 중심 |
| 정규화 rank | 18 | BS별 거리 순위를 [0,1]로 정규화 |
| Top-5 쌍별 차이 | 10 | C(5,2)=10개, 가장 가까운 5개 BS 간 거리 차이 |

**신규 피처 (86D):**

**① IRLS 신뢰도 가중치 (18D):** Range residual의 절댓값이 작을수록 해당 기지국의 측정이 현재 anchor와 일관적이다. 이를 수치화하여 각 기지국의 신뢰도 점수로 사용한다.

$$w_i = \frac{1/(|r_i| + 5)}{\sum_{j=1}^{18} 1/(|r_j| + 5)}$$

이 가중치 벡터는 IRLS(Iteratively Reweighted Least Squares)에서 가중치를 결정하는 방식과 동일한 원리이며, 기계학습 모델이 어떤 BS를 신뢰할지 판단하는 데 활용된다.

**② Sub-anchor 위치 및 불확실성 (9D):** 단계 2에서 계산한 세 sub-anchor 위치(sa4, sa6, sa8)를 피처에 포함한다(6D). 추가로, main anchor, WLS multi, IDW centroid, sa4, sa6, sa8 등 6개 추정치의 x좌표 표준편차, y좌표 표준편차, main anchor와 WLS multi 사이의 유클리드 거리를 불확실성 지표로 포함한다(3D). 여러 물리 추정치가 서로 동의할수록 현재 측위 상황이 신뢰할 만하다는 정보를 담는다.

**③ TDOA-like 쌍별 피처 (56D):** RTT 측정값 기준으로 가장 가까운 8개 BS를 선택하고, C(8,2)=28쌍에 대해 두 종류의 피처를 계산한다.

절대 차이(28D): $\delta_{ij} = \hat{d}_i - \hat{d}_j$

정규화 비율(28D): $\rho_{ij} = \dfrac{\hat{d}_i - \hat{d}_j}{\hat{d}_i + \hat{d}_j} \in [-1, 1]$

정규화 비율 $\rho_{ij}$는 두 측정값에 공통으로 포함된 NLOS bias가 분자와 분모에서 상쇄되는 효과가 있어, 순수한 상대적 거리 관계를 담는 무차원 피처이다. TDOA(Time Difference of Arrival) 기반 측위 이론에서 차분 측정이 공통 오차를 제거하는 원리를 RTT 데이터에 응용한 것이다.

**④ Sub-anchor 잔차 (3D):** $\|\hat{\mathbf{p}}_\text{sa4} - \hat{\mathbf{p}}_\text{anchor}\|_2$, $\|\hat{\mathbf{p}}_\text{sa6} - \hat{\mathbf{p}}_\text{anchor}\|_2$, $\|\hat{\mathbf{p}}_\text{sa8} - \hat{\mathbf{p}}_\text{anchor}\|_2$. 각 sub-anchor가 main anchor와 얼마나 다른지를 나타내며, 이 값이 클수록 측위 상황이 불안정함을 의미한다.

### 단계 4: OOF 5-fold Stacking (잔차 학습)

학습 목표는 anchor의 잔차 $\mathbf{r} = \mathbf{y} - \hat{\mathbf{p}}_\text{anchor}$이다. 즉, 기계학습 모델은 물리 추정의 오차를 예측하여 보정하는 역할을 한다.

**Base 모델 6종:**

| 모델 | 주요 설정 |
|---|---|
| LightGBM ① | n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=20 |
| LightGBM ② | n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=25 |
| LightGBM ③ | n_estimators=4000, learning_rate=0.01, num_leaves=127, min_child_samples=15 |
| LightGBM ④ | n_estimators=2500, learning_rate=0.02, num_leaves=63, min_child_samples=25 |
| Random Forest | n_estimators=500, max_features=sqrt, min_samples_leaf=2 |
| Ridge Regression | alpha=1.0, StandardScaler 전처리 |

5-fold Out-of-Fold(OOF) 방식으로 각 샘플의 예측을 해당 샘플이 학습에 사용되지 않은 fold의 모델로 생성한다. 이를 통해 700명 전체에 대한 OOF 예측 행렬 $(700, 12)$를 구성한다.

**Meta 모델:** OOF 예측 행렬을 입력으로 LightGBM meta 모델(n_estimators=500, num_leaves=31, min_child_samples=30)을 학습한다.

LightGBM을 기반으로 선택한 이유는 sklearn GradientBoostingRegressor 대비 leaf-wise 트리 성장 방식이 같은 학습 시간에 더 높은 표현력을 가지며, min_child_samples, reg_alpha, reg_lambda 등 세밀한 정규화 파라미터로 700개의 소규모 데이터셋에서 과적합을 효과적으로 제어할 수 있기 때문이다. 4종의 서로 다른 하이퍼파라미터 구성과 Random Forest, Ridge의 이종 모델을 혼합하여 앙상블 다양성을 확보하였다.

### 단계 5: 최종 위치 추정

$$\hat{\mathbf{p}}_\text{final} = \hat{\mathbf{p}}_\text{anchor} + f_\text{meta}([\hat{\mathbf{r}}_1, \hat{\mathbf{r}}_2, \ldots, \hat{\mathbf{r}}_6])$$

전체 데이터로 base 모델을 재학습한 후, meta 모델이 예측한 잔차를 anchor에 더하여 최종 위치를 출력한다.

---

## 3. Agent AI 활용 방안

본 프로젝트에서는 Claude Code(Anthropic)를 활용하였다. AI와 본인의 역할을 다음과 같이 구분하였다.

| 역할 | 담당 |
|---|---|
| 데이터 분포 수치 분석 (NLOS bias 통계, 거리-bias 상관) | Claude Code |
| NLOS 측위 관련 논문 검색 및 요약 (IRLS, TDOA, M-estimator) | Claude Code |
| 피처 생성 및 스태킹 코드 구현 | Claude Code |
| 비교 실험 스크립트 실행 및 수치 수집 | Claude Code |
| 알고리즘 방향 결정 (어떤 피처를 추가할지, 모델 구조) | 본인 |
| 실험 결과 해석 및 기각 판단 (LOO 일관성 기각, ratio 피처 유지) | 본인 |
| 보고서 내용 구성 및 논리 전개 | 본인 |

구체적 활용 예시: LOO(Leave-One-Out) 일관성 피처를 추가하는 아이디어를 제안하였으나, Claude Code가 실험을 실행한 결과 NLOS bias 탐지 상관계수가 기존 range residual(0.72)보다 낮았다(0.65). 이를 바탕으로 본인이 LOO 피처를 최종 알고리즘에서 제외하는 결정을 내렸다. 또한 ratio_8 피처 제거 실험에서 성능이 악화됨을 확인하고 유지하기로 결정한 것도 동일한 방식으로 진행하였다.

---

## 4. 결과 도출 & 디스커션

### 성능 결과

| 방법 | 평균 오차 | 중앙값 오차 | 90th percentile |
|---|---|---|---|
| LS 삼변측량 (baseline) | 23.20 m | 21.72 m | 33.59 m |
| Cauchy 삼변측량 (물리 단독) | 11.10 m | - | - |
| **PAANE (OOF 기준)** | **3.35 m** | **2.09 m** | **7.23 m** |
| PAANE Train set | 3.29 m | 1.96 m | 7.49 m |

### 평가 방식의 공정성

본 알고리즘의 실제 성능 추정에는 5-fold Out-of-Fold 교차검증을 사용하였다. 각 fold에서 해당 사용자는 학습에 사용되지 않은 모델로 예측되므로, 700명 전체에 대해 학습 데이터를 보지 않은 상태에서의 예측 오차가 집계된다. Train 오차(3.29 m)와 OOF 오차(3.35 m)의 차이가 0.06 m에 불과하다는 것은 과적합이 거의 없음을 의미하며, OOF 오차가 hidden test set(300명) 성능의 신뢰할 수 있는 추정치임을 나타낸다.

baseline과의 비교 공정성: 본 알고리즘(ML 기반)과 LS 삼변측량(물리 기반)의 직접 비교는 방법론 수준의 차이가 크나, 물리 기반 방법 중 가장 강력한 Cauchy 삼변측량과 비교해도 3배 이상의 오차 감소(11.10 m → 3.35 m)를 보인다. 이는 ML 보정이 단순한 복잡도 증가가 아닌 실질적 성능 향상을 제공함을 보여준다.

### 알고리즘 장점

첫째, **물리-ML 계층 구조**로 인해 ML이 물리 모델이 설명하지 못하는 NLOS 잔차를 선택적으로 보정하므로, 순수 ML 대비 소규모 데이터(700명)에서도 안정적이다.

둘째, **IRLS 신뢰도 가중치**를 피처로 포함함으로써 기계학습 모델이 각 기지국의 측정 신뢰도를 스스로 학습할 수 있다. 이는 단순히 거리 측정값만 입력하는 방식보다 더 풍부한 정보를 제공한다.

셋째, **OOF 스태킹**은 데이터 누수 없이 앙상블 효과를 극대화하며, 과적합 위험을 방법론 수준에서 차단한다.

### 알고리즘 한계 및 단점

본 알고리즘은 평균 오차 측면에서는 우수하나, 극단 NLOS 사용자(18개 BS 중 대다수가 50 m 이상 bias)에 대한 최대 오차가 약 27 m에 달한다. 이는 RTT 거리값만으로는 극심한 NLOS 환경에서 신뢰할 수 있는 기지국을 식별하는 데 구조적 한계가 있기 때문이다. 실제 채널 임펄스 응답(CIR) 등 신호 품질 정보가 주어졌다면, 기지국별 LOS/NLOS 분류 정확도를 높여 이 한계를 극복할 수 있다.

### Future Work

거리값 외에 수신 신호 강도(RSRP), 도착각(AoA) 등 추가 측정값이 제공된다면, 해당 정보를 피처에 통합하여 기지국별 신뢰도 추정 정확도를 높일 수 있다. 또한 Graph Neural Network를 적용하여 기지국 간 상호 관계를 공간 그래프로 모델링하는 방향도 유망하다.

---

## 5. Reference

- Peng, Y. et al., "On the Convergence of IRLS and Its Variants in Outlier-Robust Estimation," CVPR 2023. (IRLS의 수렴 이론적 기반 — 본 논문의 IRLS 가중치 아이디어는 해당 논문의 수렴 성질을 참고하였으며, 본 알고리즘은 IRLS를 반복 최적화에 직접 적용하는 대신 가중치를 피처로 변환하여 ML 모델에 통합하는 방식으로 차별화하였다.)

- Ke, G. et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017. (LightGBM base 모델 및 meta 모델 사용 — level-wise 대신 leaf-wise 트리 성장으로 소규모 데이터에서 더 효율적인 표현력을 제공.)

- "Outlier Rejection for 5G-Based Indoor Positioning in Ray-Tracing-Enabled Industrial Scenario," arXiv:2409.12585, 2024. (동일 5G InF 산업 시나리오에서 IRLS 기반 아웃라이어 제거 방법 — 본 알고리즘은 IRLS 반복 추정 대신 신뢰도 가중치를 피처로 변환하여 ML과 결합하는 방식으로 차별화하였다.)
