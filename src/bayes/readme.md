The goal is to do model-based reinforcement learning by modeling the transition densities $P(S' \mid S, A)$; for 
simplicity we'll suppress the dependence and just write $P(S)$.  We'll do this by combining parametric and 
nonparametric models, i.e.

$$
P(S) = P(\mathcal{M}_1) P( S \mid \mathcal{M}_1) + P( \mathcal{M}_2 ) P( S \mid \mathcal{M}_1 ),
$$

where $\mathcal{M}_1$ is a parametric model (ideally derived from some domain-relevant knowledge), and 
$\mathcal{M}_2$ is a nonparametric model, which we use in order to guard against misspecification. 
