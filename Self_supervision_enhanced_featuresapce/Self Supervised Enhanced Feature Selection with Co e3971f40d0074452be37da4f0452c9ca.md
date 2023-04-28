# Self Supervised Enhanced Feature Selection with Correlated Gates.

ICLR(2022)

[2114_self_supervision_enhanced_feat.pdf](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/2114_self_supervision_enhanced_feat.pdf)

## 1. Abstract, Introduction

다차원(High-Dimensional) 한 데이터가 많은 분야(의학, biology, healthcare 등등) 에서 각광을 받으면서, 라벨에 꼭필요할만한 중요한 변수들만 뽑는 과정이 상당히 중요해지고 있다. 

예를 들어, genetic disorders은 수십만가지의 gene expression중에서 단 일부에만 영향을 미치는 경우가 많고, 이를 찾는 과정은 쉽지않다는 것이다. 

이 과정에서 딥러닝등의 방법을 사용하게 되는데, 데이터 자체가 correlation이 매우 많거나, 라벨이 부족함 등의 문제가 많아서 좋은 feature을 select하는 과정 자체가 쉽지 않다는 것이다. 

저자는 이를 해결할 수 있는 self-supervised한 architecture을 제시한다. 

## 2. Related Work.

- Feature Selection:
    - 가장기본적인 Wrapper&filter / Lasso &Elastic 등의 Regularization 등을 통해서 feature을 selection 하는 방법
    - 이런 tabular data 에 대해서 Semi supervised관점에서 딥러닝을 통해 self-supervision 과정을 거쳐서  feature selection 을 적용한 최초의 시도라고 한다.
- Self-supervised Learning:
    - NLP나 비전에서는 이미 BERT나 RESnet 같이 pretrained된 모델을 domain에 fine tuning해서 사용하는 self-supervised learning이 사용되고 있다.
    - 하지만, tabular domain 에서는, 이런 시도들이 거의 없었고, 저자는  gate vector 라는 개념을 도입해서 (0,1 로 그 feature을 쓸지 안쓸지 정하게됨 )input vector $x$ 를  pretext tasks 에서 reconstruction하게 됨.

## 3. Problem Formulation

 $X=(X_1,...,X_p)$ 라고 하고, (p는 feature의 개수) $Y$  는 라벨의 정보를 나타내는 target outcome(disease) 라고 하면, 일반적인 feature selection 문제에서는, 

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled.png)

다음과 같은 Loss function을 통해 문제를 해결하고자 한다. 하지만, 위와같은 minimization problem은 적당한 s, 위 식에서는 $\delta$ 의 값을 찾는것이 exponential 한 시간이 걸리기 때문에, 어렵게 된다. 

따라서 이 논문에서는 해당문제를 possibly correlate binary random variables  의 space로 바꾸어서, 문제를 풀게 되고, 변수들간의 independence를 가정하지 않는다. 

$M=(M_1,...,M_p) \in$  {$0,1$} 은 binary random variable이라고 가정한다. 이 random variable은 특정 distribution $p_M$ 에의 해서 생성(결정) 이 된다. 그리고 이것의 realization $m$ 은 gate vector 라는 것을 통해  특정 변수를 select할지를 결정하게 된다.  이를 수식으로 나타내보면

$$
\tilde{x}=m\odot x+ (1-m) \odot \bar{x}
$$

여기서 $x$ 는 원래 input이고 $m$ 은 선택하면 1 선택하지 않으면 0을 나타내는 변수, $\bar{x}$  는 $x$들의 평균을 사용하는데, 사용되지 않았을때도 특정한 의미를 가지게끔 하기 위해서 이런 방법을 사용한다고 한다. 

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%201.png)

이를 종합해서 gate operation이 진행된 $\tilde{x}$ 를 통해 encoder $f$를 훈련시키고, 훈련 된 encoder 을 통해 prediction을 진행하고 실제 y와의 loss 를 최소화하는 것이다. 

여기서 $\beta$ 는 balancing coefficeint 를 나타내고, 몇개의 feature selection을 진행할지를 결정할 coefficient 라고 생각하면 된다. 

## 4. Method Structure

변수들간의 inter-correlation 혹은 multicollinearity 문제와 unlabled된 sample이 많은 상황을 해결하기 위해 저자는 다음과 같은 two-step training procedure기반의  semi-supervised 구조를 제안한다.

해당구조는 특정 변수를 선택할지 말지 정하는 selection probabilty  $\pi$ 와 모델을 latent 한 dimension으로  encode하는 encoder $f_{\theta}$, encode된 모델 기반으로 predict하는 predictor $f_{\phi}$ 를 훈련시키는 모델이다.  

- Self Supervision Phase: encoder 역할을 하는 $f_{\theta}$ 를 unlabeled sample들을 통해서 구해낸다.
- Supervision Phase: learned representation을 통해 predict 를 하고 위 세가지 변수들을 업데이트 한다.

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%202.png)

## 4-1 Multivariate Bernoulli Gate Vectors using Gaussina Copula

저자는  correlation을 고려한 gate vector $m$ 을 construct하기 위해서 Gaussian Copula 를 이용하게 된다.  Multivarite한 Cumulative distribution function CDF를 통해서 multivariate Gaussian Distribution 을 만들고(mean 0, correlation matrix R ) 여기서 R은 원래 데이터들의 correlation기반으로 만든다.

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%203.png)

이렇게 p개의 feature에[ 대한 가우시안 Copula를 multivarite 한 상황에 만들고,  

만들어진 Copula에서 gate vector m을 generate함으로써 원래 input feature들의 correlation (generate하는 과정은 아래 설명)

structure을 고려하면서 베르누이 분포를 만들어 내고, 결과적으로correlation이 고려된  gate vector $m$ 을 만들어 낼 수 잇게 된다. 

( $m_k=1 \ if\ U_k\leq\pi_k$, $m_k=0\ if \ U_k>\pi_k$) , $U_i\in[0,1]$ 인 Uniform distirbution. 

## 4-2 Self Supervision Phase

Self supervision과정은 보통 unlabeledd sample을 사용하여 구조를 학습하는데 사용되고 

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%204.png)

해당 논문에서 제안한 모델이 self supervision phase에서 학습하는 부분은 위와 같습니다.

Encoder: x를 reconstruct하는 과정을 통해서 latent reprezentation z를 만들어냄과 동시에 그 때사용되는 parameter $\theta$를 학습합니다.

feature vector estimator: latent stapce에서 다시 원래대로 복구해주는  

gate vector estimator: z as a input and outputs vecotr where prediction of which features have been selected. 

 $\tilde{x}$ 를 만들어낼 때,  masking(그 변수를 넣을지 말지를 결정하는 확률) 은 COpula 에의해 Correlation이 고려된 베르누이 랜덤분포를 통해서 생성이되고 \

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%205.png)

  ( $m_k=1 \ if\ U_k\leq\pi_k$, $m_k=0\ if \ U_k>\pi_k$) ,이렇게 p(m_k)가 continuous하지 않으므로 continual하게 relaxation하는 reparametrization trick을 사용하여, $\pi$에 대하여 미분 가능한 식으로 만들어준다. 

이후 촐레스키 디컴포지션을 이용하여, 변수마다(correlation이 고려된) 유니폼한 r.v를 생성하고, mk를 구할 수 잇게 된다. 

결과적으로,

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%206.png)

다음과 같은 식을 minimize하는 상황을 보여줍니다. 

## 4-3 Supervision Phase

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%207.png)

이제 생성된 latent space를 가지고, predict하는 supervison phawse를 결정하ㅔㄱ 되는데,

![Untitled](Self%20Supervised%20Enhanced%20Feature%20Selection%20with%20Co%20e3971f40d0074452be37da4f0452c9ca/Untitled%208.png)

결과적으로 다음과 같은 loss를 minimize하는 문제가 됩니다. 

## 5. 성능

SEFS 방식이 상당히 잘작동했다. 특히 라벨이 없고 Correlation이 많으면서, 몇가지 impact 있는 feature들을 찾는데에 있어서