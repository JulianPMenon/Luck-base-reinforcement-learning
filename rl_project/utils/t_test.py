import numpy as np
from scipy.stats import ttest_ind

def run_ttest(sample1, sample2):
    """
    Run an independent two-sample t-test and print the t-score and p-value.
    Args:
        sample1 (list or np.ndarray): First sample of values.
        sample2 (list or np.ndarray): Second sample of values.
    """
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=False)
    print(f"T-score: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    return t_stat, p_value

if __name__ == "__main__":
    # Example usage: replace with your actual values
    contrastive_rewards_easy = [0.58379296875, 0.85240234375, 0.881296875, 0.8409453125, 0.8938281249999999, 0.8621718749999999, 0.92086328125, 0.8471132812500001, 0.8621914062500001, 0.8708203125]
    rnd_rewards_easy = [0.3303120712881295, 0.321955185021822, 0.293153796097857, 0.3148656880099485, 0.5197415465489542, 0.32743105066980277, 0.31249930440568585, 0.33574875852922487, 0.2656215504065652, 0.3337078812819584]  # Replace with your RND values
    contrastive_no_her_easy = [
    0.47651953124999996,
    0.7892031250000001,
    0.4857421874999999,
    0.4296875,
    0.22595741421568627,
    0.16352634803921565,
    0.40701171875,
    0.2138020833333333,
    0.26204044117647063,
    0.405032169117647
]
    contrastive_rewards_hard = [
    0.09070408163265306,
    0.0738265306122449,
    0.10414165666266507,
    0.1390612244897959,
    0.13887755102040816,
    0.13744387755102042,
    0.1800714285714286,
    0.15360144057623049,
    0.13112755102040816,
    0.1794017607042817
]
    rnd_rewards_hard = [
    0.04062361889985764,
    0.0245351202203453,
    0.025054560911720605,
    0.013594803718406658,
    0.029217004043762138,
    0.032040020990970676,
    0.030416024416111295,
    0.03479918612406467,
    0.04651258694524716,
    0.031171916581403565
]
    print("Running t-tests for easy task...")
    run_ttest(contrastive_rewards_easy, rnd_rewards_easy)
    print("Running t-tests for contrastive no HER easy task...")
    run_ttest(contrastive_no_her_easy, rnd_rewards_easy)
    print("Running t-tests for hard task...")
    run_ttest(contrastive_rewards_hard, rnd_rewards_hard)
    
### Easy Task T-test Results    
### T-score: 13.8151
### P-value: 0.0000
###  t-tests for contrastive no HER easy task
### T-score: 0.8127
### P-value: 0.4330
###  t-tests for hard task
### T-score: 8.9323
### P-value: 0.0000
