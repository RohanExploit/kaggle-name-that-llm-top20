import os
import glob
import pandas as pd

files = glob.glob(r"E:\projects\kaggle\outputs\name-that-llm-publicleaderboard-*.csv")
if not files:
    print("No leaderboard file found.")
    exit(0)

latest = sorted(files)[-1]
df = pd.read_csv(latest)
print(f"Leaderboard: {os.path.basename(latest)}")
print(f"Total Teams: {len(df)}")
print("-" * 60)
print(f"{'Rank':<5} | {'Score':<7} | {'Team Name'}")
print("-" * 60)

for _, r in df.head(30).iterrows():
    name = str(r['TeamName']).encode('ascii', 'replace').decode('ascii')
    print(f"{int(r['Rank']):<5} | {float(r['Score']):<7.4f} | {name}")

print("-" * 60)
# Look for user team
for _, r in df.iterrows():
    name = str(r['TeamName']).lower()
    if 'rohan' in name or 'exploit' in name:
        print(f"USER TEAM FOUND: Rank {int(r['Rank'])} | Score {float(r['Score']):.4f} | {r['TeamName']}")
