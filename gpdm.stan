
data {

  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> H;

  int<lower=1,upper=J> movie[N];
  int<lower=1,upper=5> rating[N];

  matrix[J,4] Q;

  matrix[H,4] alpha_class;
}

parameters {

  simplex[H] pi;

  vector[J] beta;

  vector<lower=0>[4] lambda;

  ordered[4] c;

}

model {

  beta ~ normal(0,1);

  lambda ~ lognormal(0,0.5);

  for(n in 1:N){

    vector[H] lp;

    for(h in 1:H){

      real eta;

      eta =
        beta[movie[n]]
        +
        Q[movie[n],1] * lambda[1] * alpha_class[h,1]
        +
        Q[movie[n],2] * lambda[2] * alpha_class[h,2]
        +
        Q[movie[n],3] * lambda[3] * alpha_class[h,3]
        +
        Q[movie[n],4] * lambda[4] * alpha_class[h,4];

      lp[h] =
        log(pi[h]) +
        ordered_logistic_lpmf(
          rating[n] | eta , c
        );
    }

    target += log_sum_exp(lp);
  }
}

