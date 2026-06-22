import pandas as pd
from scipy import stats
import math

def proportions_ztest(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1+k2)/(n1+n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z = (p1-p2)/se if se > 0 else 0.0
    p = 2*(1-stats.norm.cdf(abs(z)))
    return z, p

base = pd.read_csv("results/musique-natural/interplay_analysis/interplay_summary_nemotron_base_100q.csv")
tuned = pd.read_csv("results/musique-natural/interplay_analysis_tuned_val100/interplay_summary.csv")

for label, df in [("base", base), ("tuned", tuned)]:
    uncertain = df[df["semantic_entropy"] > 0]
    certain   = df[df["semantic_entropy"] == 0]
    # M = uncertain + not searched
    M  = uncertain[~uncertain["searched"]]
    # CP = certain + not searched
    CP = certain[~certain["searched"]]
    # OS = certain + searched (over-search)
    OS = certain[certain["searched"]]
    # WC = uncertain + searched
    WC = uncertain[uncertain["searched"]]
    total = len(df)
    print(f"\n{'='*40}")
    print(f"  {label.upper()} ({total} hops, {df['example_id'].nunique()} questions)")
    print(f"  Uncertain hops : {len(uncertain)} ({len(uncertain)/total:.1%})")
    print(f"  M  (missed)    : {len(M)}  ({len(M)/len(uncertain):.1%} of uncertain)")
    print(f"  WC (searched)  : {len(WC)} ({len(WC)/len(uncertain):.1%} of uncertain)")
    print(f"  CP (certain+skip): {len(CP)} ({len(CP)/total:.1%})")
    print(f"  OS (over-search) : {len(OS)} ({len(OS)/total:.1%})")

# Significance test on M-rate (missed / uncertain)
b_unc = base[base["semantic_entropy"] > 0]
t_unc = tuned[tuned["semantic_entropy"] > 0]
b_missed = (~b_unc["searched"]).sum()
t_missed = (~t_unc["searched"]).sum()
b_missed = (~b_unc["searched"]).sum()
t_missed = (~t_unc["searched"]).sum()
z, p = proportions_ztest(b_missed, len(b_unc), t_missed, len(t_unc))
print(f"\nM-rate: base={b_missed}/{len(b_unc)} ({b_missed/len(b_unc):.1%})  tuned={t_missed}/{len(t_unc)} ({t_missed/len(t_unc):.1%})")
print(f"Two-proportion z-test: z={z:.3f}, p={p:.4f} {'*' if p<0.05 else '(n.s.)'}")

# Also test OS-rate (over-search)
b_cert = base[base["semantic_entropy"] == 0]
t_cert = tuned[tuned["semantic_entropy"] == 0]
b_os = b_cert["searched"].sum()
t_os = t_cert["searched"].sum()
z2, p2 = proportions_ztest(b_os, len(b_cert), t_os, len(t_cert))
print(f"\nOS-rate: base={b_os}/{len(b_cert)} ({b_os/len(b_cert):.1%})  tuned={t_os}/{len(t_cert)} ({t_os/len(t_cert):.1%})")
print(f"Two-proportion z-test: z={z2:.3f}, p={p2:.4f} {'*' if p2<0.05 else '(n.s.)'}")