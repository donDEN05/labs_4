"""Cooperative discount simulator. Demand response and cost are assumptions."""
import numpy as np


class DiscountEnv:
    def __init__(self,features,prices,budget=5.,cost_ratio=.6,minimum_margin=.1,elasticity=1.2,substitution=.25,regime='normal'):
        self.features=np.asarray(features,dtype=np.float32)
        self.prices=np.asarray(prices,float)
        self.n=len(prices)
        self.budget=float(budget)
        if self.features.ndim!=3 or self.features.shape[2]!=self.n or not self.features.size or not np.isfinite(self.features).all() or ((self.features<0)|(self.features>1)).any() or not np.isfinite(self.prices).all() or (self.prices<=0).any() or not np.isfinite(self.budget) or self.budget<0:
            raise ValueError('finite probabilities, positive prices and nonnegative budget required')
        self.cost_ratio=cost_ratio
        self.minimum_margin=minimum_margin
        self.elasticity=elasticity
        self.substitution=substitution
        self.regime=regime
        if not np.isfinite([cost_ratio,minimum_margin,elasticity,substitution]).all() or cost_ratio<0 or minimum_margin<0 or max(cost_ratio,self.cost())+minimum_margin>1+1e-9 or elasticity<0 or substitution<0:
            raise ValueError('finite nonnegative parameters and feasible minimum margin required')
        self.grid=np.array([0,.05,.1,.15,.2])
        self.reset(11)

    def reset(self,seed):
        self.rng=np.random.default_rng(seed)
        self.person=int(self.rng.integers(len(self.features)))
        self.t=0; self.remaining=self.budget; self.market=1.
        return self.observe()

    def observe(self):
        p=self.features[self.person,min(self.t,self.features.shape[1]-1)]
        scalars=np.column_stack([p,np.full(self.n,self.remaining/max(self.budget,1)),np.full(self.n,self.t/self.features.shape[1]),np.full(self.n,self.market)])
        return np.concatenate([scalars,np.eye(self.n)],axis=1).astype(np.float32)

    def cost(self):
        return min(.85,self.cost_ratio+.15) if self.regime=='high_cost' else self.cost_ratio

    def project(self,actions,remaining=None):
        a=np.asarray(actions)
        if a.shape!=(self.n,) or not np.isfinite(a).all() or (a<0).any() or (a>4).any() or not np.equal(a,np.floor(a)).all():
            raise ValueError('one discrete action 0..4 per category required')
        budget=self.remaining if remaining is None else remaining
        if not np.isfinite(budget) or budget<0:raise ValueError('budget must be finite and nonnegative')
        cost=self.cost()
        max_discount=max(0,1-cost-self.minimum_margin)
        d=np.minimum(self.grid[a.astype(int)],np.floor((max_discount+1e-9)/.05)*.05)
        # ponytail: conservative one-unit-per-category budget; extend projection if multi-unit demand is introduced.
        while float(d@self.prices)>budget+1e-9:
            i=int(np.argmax(d*self.prices))
            d[i]=max(0,round(d[i]-.05,8))
        return d

    def step(self,actions):
        d=self.project(actions)
        proposed=self.grid[np.asarray(actions,int)]
        proposed_violations=int(proposed@self.prices>self.remaining+1e-8)+int(((1-proposed-self.cost())<self.minimum_margin-1e-8).sum())
        base=self.features[self.person,self.t].copy()
        elasticity=self.elasticity
        cost=self.cost()
        if self.regime=='low_demand': base*=.5
        if self.regime=='low_elasticity': elasticity*=.25
        if self.regime=='markov_switching':
            # Transition P=[[.85,.15],[.30,.70]], states normal/low demand.
            low=self.market<1
            low=self.rng.random() < (.70 if low else .15)
            self.market=.5 if low else 1.
            base*=self.market
        probability=np.clip(base+elasticity*d-self.substitution*(d.sum()-d)/max(self.n-1,1),.001,.999)
        buys=(self.rng.random(self.n)<probability).astype(float)
        expense=float((buys*d)@self.prices)
        revenue=float((buys*(1-d))@self.prices)
        profit=revenue-float(buys@(cost*self.prices))
        previous_budget=self.remaining
        self.remaining=max(0,self.remaining-expense)
        min_margin=float(np.min(1-d-cost))
        violations=int(expense>previous_budget+1e-8)+int(min_margin<self.minimum_margin-1e-8)
        info=dict(profit=profit,revenue=revenue,discount_cost=expense,turnover=float(buys.sum()),proposed_violations=proposed_violations,applied_violations=violations,minimum_margin=min_margin,remaining=self.remaining,discounts=d.tolist(),p=probability.tolist(),actions=np.asarray(actions).tolist())
        reward=profit/10-.05*proposed_violations
        self.t+=1
        done=self.t==self.features.shape[1]
        return self.observe(),float(reward),done,info
