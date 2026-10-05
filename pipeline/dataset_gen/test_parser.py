import re

def clean_gsm_solution(ans):
    # remove <<...>> calculation tags
    ans = re.sub(r'<<.*?>>', '', ans)
    lines = [l.strip() for l in ans.split('\n') if l.strip()]
    return lines

print("Testing cleaner")
