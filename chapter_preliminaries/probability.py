import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
# a = torch.ones((2, 3))
# print(a)
print(fair_probs)
multinomial.Multinomial(1, fair_probs).sample()
print(multinomial.Multinomial(1, fair_probs).sample())

print(multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts)
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)


