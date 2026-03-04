import pandas as pd
from features import load_matches_csv

matches = load_matches_csv('data/spi_matches.csv')
cutoff = pd.Timestamp('2025-03-02')
test = matches[matches['date'] >= cutoff]

print(f"Test matches: {len(test)}")
test['score'] = test['home_goals'].astype(str) + '-' + test['away_goals'].astype(str)
score_counts = test['score'].value_counts()
print(f"\nActual outcome frequencies (top 10):")
for score, count in score_counts.head(10).items():
    pct = 100 * count / len(test)
    print(f"  {score}: {count} ({pct:.1f}%)")

# 1-1 specific
ones_1 = len(test[test['score'] == '1-1'])
ones_1_pct = 100 * ones_1 / len(test)
print(f"\n1-1 actual: {ones_1} matches ({ones_1_pct:.1f}%)")
