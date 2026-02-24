---
description: how to deploy code changes to the DGX server
---

// turbo-all

## Rule: Fix Locally First, Push Once at the End

1. Make ALL code fixes locally in `/Users/aaronr/Downloads/svamitva_model/`
2. Test/verify locally
3. When everything is ready, push to BOTH Git and DGX in a single step

## Credentials
- **Host:** `192.168.6.21`
- **User:** `sods.user04`
- **Remote base:** `/jupyter/sods.user04/svamitva_model/`

## Key Paths on DGX
| Purpose | Path |
|---|---|
| Code | `/jupyter/sods.user04/svamitva_model/` |
| Data | `/jupyter/sods.user04/DATA/` |
| Checkpoints | `/jupyter/sods.user04/svamitva_model/checkpoints/` |

## Final Push (do this ONCE after all fixes are ready)

### Step 1: Push to GitHub
```bash
git add -A && git commit -m "message" && git push
```

### Step 2: Upload changed files directly to DGX via SSH (NEVER git pull)
For each changed Python file:
```bash
ssh sods.user04@192.168.6.21 "cat > /jupyter/sods.user04/svamitva_model/<path/to/file.py>" << 'EOF'
<full file content here>
EOF
```

For the notebook:
```bash
ssh sods.user04@192.168.6.21 "cat > /jupyter/sods.user04/svamitva_model/SVAMITVA_Final.ipynb" < SVAMITVA_Final.ipynb
```

### Step 3: Verify on DGX
```bash
ssh sods.user04@192.168.6.21 "grep -c 'KEYWORD' /jupyter/sods.user04/svamitva_model/path/to/file.py"
```
