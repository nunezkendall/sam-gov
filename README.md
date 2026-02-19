# SAM.gov Daily Opportunity Pull + Scoring

This script pulls new/updated SAM.gov Contract Opportunities for selected NAICS codes and Small Business set-asides, filters by keywords and dollar range, outputs a clean JSON list, then scores and ranks the top opportunities using the OpenAI API.

**Files**
- `sam_daily.py` main script
- `config.example.json` configuration template

## Setup

1. Create `config.json` based on `config.example.json` and adjust if needed.
2. Set environment variables:
   - `SAM_API_KEY` (required)
   - `OPENAI_API_KEY` (optional, required for scoring)
3. Install dependencies:
   - `pip install requests`

## Run Manually

```powershell
python sam_daily.py
```

Skip scoring:

```powershell
python sam_daily.py --no-score
```

## Output

- `output/opportunities_raw.json` raw API data (deduped)
- `output/opportunities_filtered.json` clean, filtered list
- `output/opportunities_scored.json` OpenAI scoring output
- `output/top10.json` ranked top 10 pursue list
- `data/state.json` last run timestamp

## Windows Task Scheduler (Daily)

1. Open **Task Scheduler** -> **Create Task**.
2. **General** tab:
   - Name: `SAM.gov Daily Pull`
   - Run whether user is logged on or not
3. **Triggers** tab:
   - New -> Daily -> set time
4. **Actions** tab:
   - New -> Start a program
   - Program/script: `python`
   - Add arguments: `sam_daily.py`
   - Start in: `C:\Users\nunez\OneDrive\Documents\NLTC\Sam-Gov`
5. **Conditions** / **Settings** tabs: adjust as needed.

If the task needs environment variables, you can either:
- Add them to the system/user environment variables, or
- Use a wrapper script that sets them before calling Python.

## GitHub Actions (Daily)

This repo includes a workflow at `.github/workflows/daily-sam.yml` that runs daily at **10:00 AM MST** (17:00 UTC).

1. Create a GitHub repo and push this folder.
2. In GitHub, go to **Settings → Secrets and variables → Actions** and add:
   - `SAM_API_KEY`
   - `OPENAI_API_KEY` (optional; required for scoring)
3. The workflow will commit updated `output/` and `data/state.json` each run.

## Notes

- The script tracks both posted and updated items. It requests recent postings and then filters by last run or a fallback updated window.
- Amount filtering is strict when a value is available; unknown amounts are kept unless `require_amount` is true.
- Keyword matching is done against the title and the description text.
- Full description text is included in scoring payloads (configurable).
