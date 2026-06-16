# עבודה על Athena — מדריך גנרי

מדריך מקיף לעבודה על קלאסטר Athena של הטכניון (SLURM). מתאים לכל פרויקט.

---

## 1. מידע על הקלאסטר

### גישה
- **כתובת**: `athena.technion.ac.il`
- **התחברות**: `ssh <username>@athena.technion.ac.il`
- **העברת קבצים**: `scp` / `rsync` (ראה סעיף 9)

### Filesystem
| נתיב | תפקיד | מכסה |
|------|-------|------|
| `/home/<user>/` | תיקיית בית — קונפיגורציה, נקודות-mount קטנות | קטנה — בדוק עם `quota -s` |
| `/rg/<lab_prj>/<user>/` | מקום עבודה ראשי — קוד, נתונים, מודלים | מכסה לפי המעבדה — בדוק עם `df -h /rg/<lab_prj>/` |
| `/tmp/` | זמני, מקומי לנוד | זמני — נמחק אחרי המשימה |

**בדיקת מקום פנוי:**
```bash
df -h /rg/<lab_prj>/<user>/
du -sh /rg/<lab_prj>/<user>/*    # לפי תת-תיקייה
quota -s                          # מכסת home
```

### חומרה (Partitions / GPU)

רשימת ה-partitions, סוגי ה-GPUs וכמות ה-VRAM משתנים עם הזמן. **תמיד בדוק במצב חי**:

```bash
# רשימת כל ה-partitions וזמינות
sinfo

# פירוט על partition ספציפי
sinfo -p <partition_name> -o "%P %g %D %t %N"

# פירוט מלא של partition כולל מגבלות זמן/משאבים
scontrol show partition <partition_name>

# סוגי GPU בכל נוד (כשבדיקת VRAM/דגם נדרשת)
sinfo -o "%n %P %G" | sort -u
scontrol show node <node_name>   # פירוט נוד ספציפי כולל gres

# בנוד מחושב (אחרי srun): פרטי ה-GPU
nvidia-smi
nvidia-smi --query-gpu=name,memory.total --format=csv
```

> **טיפ**: כשבוחרים partition, לוקחים בחשבון: זמינות (`sinfo`), VRAM של ה-GPU מול גודל המודל, ומדיניות התור (shared vs. dedicated).

### QoS (Quality of Service)

QoS מגדיר זמן ריצה מקסימלי, מספר GPUs מקסימלי, וקוואטות אחרות. **רשימת ה-QoS משתנה** — תמיד בדוק חי:

```bash
# כל ה-QoS במערכת
sacctmgr show qos format=Name,MaxWall,MaxTRES,MaxJobsPU,Priority

# ה-QoS שמותרים למשתמש שלך
sacctmgr show user $USER withassoc format=User,Account,QOS

# QoS המקושרים ל-partition ספציפי
scontrol show partition <partition_name> | grep -i qos
```

הקונבנציה לרוב מקודדת בשם (לדוגמה `<duration>_<max_gpus>` כמו `24h_4g`), אבל אל תסמוך על שמות — בדוק עם `sacctmgr` את המגבלות בפועל.

---

## 2. Setup ראשוני של פרויקט

### העתקת קוד לאתנה
```bash
ssh athena.technion.ac.il
cd /rg/<lab_prj>/<user>/
git clone <repo_url>
cd <project>
```

### יצירת סביבת Python
**venv (מומלץ לפרויקטים פשוטים):**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**conda/miniconda:**
```bash
# התקנה חד-פעמית
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /rg/<lab_prj>/<user>/miniconda3
source /rg/<lab_prj>/<user>/miniconda3/bin/activate
conda init bash

# יצירת סביבה
conda create -n myenv python=3.11 -y
conda activate myenv
pip install -r requirements.txt
```

**uv (מהיר יותר):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Environment modules (אם בשימוש)
```bash
module avail                    # מודולים זמינים
module load cuda/12.1           # טעינת CUDA
module load python/3.11
module list                     # מה טעון כרגע
module purge                    # ניקוי
```

### HuggingFace cache (חשוב!)
ה-cache ברירת המחדל ב-`~/.cache/huggingface` יתפוצץ. הפנה אותו ל-`/rg/`:

```bash
# ב-~/.bashrc או בכל סקריפט SLURM
export HF_HOME=/rg/<lab_prj>/<user>/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
```

---

## 3. הרצה אינטראקטיבית

לדיבאג, ניסויים קצרים, או בדיקה לפני הגשה ל-SLURM.

> **שים לב**: בכל הדוגמאות מטה מופיעים `<partition>` ו-`<qos>` כ-placeholders — מלא לפי הפלט של `sinfo` ו-`sacctmgr show qos` (סעיף 1).

### Interactive shell עם GPU
```bash
# הקצאה אינטראקטיבית — נכנסים לנוד עם GPU
srun --partition=<partition> --qos=<qos> --gres=gpu:1 \
     --mem=32G --cpus-per-task=8 --time=2:00:00 --pty bash

# בתוך הסשן:
nvidia-smi                      # אימות GPU + פרטי VRAM
source .venv/bin/activate
python my_script.py
```

### Interactive עם x11/jupyter
```bash
# Jupyter על נוד מחושב
srun --partition=<partition> --qos=<qos> --gres=gpu:1 --time=2:00:00 --pty bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# במחשב המקומי:
ssh -L 8888:<compute_node>:8888 athena.technion.ac.il
# פתח דפדפן: http://localhost:8888
```

### Tmux/screen (לסשנים ארוכים)
```bash
tmux new -s mywork              # סשן חדש
# Ctrl+B then D — ניתוק
tmux ls                         # רשימת סשנים
tmux attach -t mywork           # חיבור מחדש
```

---

## 4. הרצה ב-batch דרך SLURM

### תבנית SLURM בסיסית
```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --time=4-00:00:00              # D-HH:MM:SS (חייב להיות ≤ MaxWall של ה-QoS)
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/myjob_%j.out
#SBATCH --error=logs/myjob_%j.err

cd /rg/<lab_prj>/<user>/<project>
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# HF cache redirect
export HF_HOME=/rg/<lab_prj>/<user>/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

python my_experiment.py --arg1 value1
```

הגשה:
```bash
mkdir -p logs
sbatch my_job.sh
# Submitted batch job 12345
```

### Job arrays (הרצה מקבילית של אותה משימה על קלטים שונים)
```bash
#!/bin/bash
#SBATCH --job-name=array_demo
#SBATCH --array=0-9                    # 10 משימות (אינדקסים 0-9)
#SBATCH --array=0-99%10                # 100 משימות, רק 10 רצות בו-זמנית
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --gres=gpu:1
#SBATCH --output=logs/arr_%A_%a.out    # %A=job_id, %a=array_index
#SBATCH --error=logs/arr_%A_%a.err

TASK_ID=$SLURM_ARRAY_TASK_ID

# קריאת פרמטרים מקובץ TSV/JSON
PARAMS=$(sed -n "$((TASK_ID+1))p" inputs.tsv)
python my_script.py --params "$PARAMS"

# או מ-JSON:
ARG=$(python -c "import json; t=json.load(open('tasks.json'))[$TASK_ID]; print(t['arg'])")
python my_script.py --arg "$ARG"
```

### תבנית מתקדמת (עם cleanup, ניהול שגיאות)
```bash
#!/bin/bash
#SBATCH --job-name=robust_job
#SBATCH --array=0-49%5
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

set -uo pipefail
trap 'echo "FAILED on line $LINENO"; exit 1' ERR

cd /rg/<lab_prj>/<user>/<project>
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/rg/<lab_prj>/<user>/hf_cache

TASK_ID=$SLURM_ARRAY_TASK_ID
WORK_DIR=/tmp/job_${SLURM_JOB_ID}_${TASK_ID}
mkdir -p "$WORK_DIR"

cleanup() {
    echo "--- Cleanup ---"
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "=== Task $TASK_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv

python my_script.py --task-id "$TASK_ID" --work-dir "$WORK_DIR"

echo "=== Done at $(date) ==="
```

### דגלי SBATCH חשובים
| דגל | תיאור |
|-----|-------|
| `--job-name=NAME` | שם להצגה ב-squeue |
| `--partition=P` | partition של החומרה |
| `--qos=Q` | QoS (מגביל זמן/GPUs) |
| `--time=D-HH:MM:SS` | זמן מקסימלי (חייב להיות ≤ QoS) |
| `--mem=32G` | זיכרון RAM |
| `--gres=gpu:N` או `--gres=gpu:<type>:N` | מספר/סוג GPUs (סוגים זמינים: `sinfo -o "%n %G"`) |
| `--cpus-per-task=N` | CPU cores |
| `--ntasks=N` | מספר משימות מקביליות (לרוב 1) |
| `--nodes=N` | מספר נודים |
| `--array=0-99[%K]` | job array; `%K` = מקבילית מקס |
| `--output=FILE` | stdout (`%j`=job id, `%A`=array job, `%a`=array idx) |
| `--error=FILE` | stderr |
| `--dependency=afterok:JOB` | המתן עד שמשימה מסתיימת בהצלחה |
| `--mail-type=END,FAIL` + `--mail-user=...` | התראות במייל |

### הגשה עם פרמטרים דינמיים (ללא script נפרד)
```bash
sbatch --job-name=quick --partition=<partition> --qos=<qos> \
       --gres=gpu:1 --mem=16G --time=2:00:00 \
       --output=logs/quick_%j.out \
       --wrap="cd /rg/.../proj && source .venv/bin/activate && python script.py"
```

---

## 5. ניטור משימות

```bash
# המשימות שלי
squeue -u $USER
squeue -u $USER --start                # זמן התחלה משוער

# פירוט על משימה ספציפית
scontrol show job <JOB_ID>

# היסטוריה (כולל משימות שהסתיימו)
sacct -u $USER --starttime now-1days
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize

# ביטול
scancel <JOB_ID>
scancel -u $USER                        # כל המשימות שלי (זהירות!)
scancel -n myjob                        # לפי שם
scancel <JOB_ID>_<ARRAY_IDX>            # array task ספציפי

# מצב הקלאסטר
sinfo                                   # partitions
squeue                                  # כל הקיו
sshare -u $USER                         # fair-share priority
```

### מעקב אחר log חי
```bash
tail -f logs/myjob_<JOB_ID>.out
tail -f logs/arr_<JOB_ID>_<ARRAY_IDX>.out
```

### בדיקת שימוש משאבים אחרי סיום
```bash
seff <JOB_ID>                           # Job efficiency report (אם מותקן)
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,State,ExitCode
```

---

## 6. ניהול נתונים ומודלים

### הורדת מודלי HuggingFace
```bash
# על נוד מחושב (לא על login)
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /rg/<lab_prj>/<user>/models/Llama-3.2-1B
# או דרך Python — אותו דבר אבל מאוחסן ב-HF_HOME
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')"
```

### Login ל-HF (לפעם הראשונה)
```bash
huggingface-cli login          # נדרש token עם permissions קריאה
# או export HF_TOKEN=... ב-.bashrc
```

### גיבוי / סנכרון נתונים
```bash
# בתוך אתנה
cp -r /rg/.../source /rg/.../backup
rsync -av /rg/.../source/ /rg/.../backup/    # incremental

# בדיקת שלמות
md5sum file.csv
sha256sum file.csv
```

---

## 7. תהליך עבודה מומלץ (development loop)

### Pattern קלאסי: edit locally → deploy → run
```bash
# 1. עריכה מקומית
vim experiments/my_script.py

# 2. commit + push
git add -A && git commit -m "feat: new analysis" && git push

# 3. pull על אתנה
ssh athena.technion.ac.il "cd /rg/<lab_prj>/<user>/<proj> && git pull"

# 4. הגשה
ssh athena.technion.ac.il "cd /rg/.../proj && sbatch scripts/my_job.sh"
```

### Pattern חלופי: rsync ישיר (לקבצים שלא ב-git)
```bash
# מהמכונה המקומית
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
      ./my_project/ athena.technion.ac.il:/rg/<lab_prj>/<user>/my_project/
```

> **המלצה**: עדיף git על rsync — מבטיח reproducibility וזיהוי שינויים.

---

## 8. העברת קבצים

```bash
# הורדה מאתנה
scp athena.technion.ac.il:/path/to/file.csv ./
scp -r athena.technion.ac.il:/path/to/dir ./

# העלאה לאתנה
scp local_file.py athena.technion.ac.il:/rg/.../

# סנכרון תיקיות (יעיל יותר עבור הרבה קבצים)
rsync -av --progress athena.technion.ac.il:/rg/.../results/ ./local_results/
rsync -av --progress --exclude='*.tmp' ./data/ athena.technion.ac.il:/rg/.../data/

# פתיחת SSH ב-multiplexing (חיבור מהיר יותר לפקודות חוזרות)
# הוסף ל-~/.ssh/config:
cat >> ~/.ssh/config <<EOF
Host athena
    HostName athena.technion.ac.il
    User <username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
EOF
# עכשיו: ssh athena ; scp athena:... .
```

---

## 9. ~/.ssh/config מומלץ

```ssh-config
Host athena
    HostName athena.technion.ac.il
    User <username>
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m

Host athena-tunnel
    HostName athena.technion.ac.il
    User <username>
    LocalForward 8888 localhost:8888    # Jupyter
    LocalForward 8501 localhost:8501    # Streamlit
```

---

## 10. פתרון בעיות נפוצות

| בעיה | פתרון |
|------|-------|
| `srun: error: Unable to allocate resources` | partition/qos לא תואמים, או הקלאסטר עמוס — `sinfo` |
| `CUDA out of memory` | בחר partition עם GPU גדול יותר (`sinfo -o "%n %P %G"`), או הקטן batch_size |
| משימה תקועה ב-`PD` (Pending) | בדוק עם `squeue -u $USER --start` או `scontrol show job <ID>` |
| `Reason=QOSMaxJobsPerUserLimit` | יש לך יותר מדי משימות פעילות — חכה / scancel חלק |
| `Reason=AssocGrpCPURunMinutesLimit` | חרגת מ-CPU-minutes שמוקצים — חכה / קצר ל-`--time` |
| Disk quota exceeded | `du -sh /rg/<lab>/$USER/* \| sort -h` — מצא תיקיות גדולות |
| `Permission denied (publickey)` | הוסף `~/.ssh/id_*.pub` ל-`authorized_keys` באתנה |
| מודל HF לא מוריד | `huggingface-cli login` או הגדר `HF_TOKEN` |
| `ModuleNotFoundError` בסשן SLURM | חסר `source .venv/bin/activate` בסקריפט |
| לוג ריק | חסר `export PYTHONUNBUFFERED=1` או הפלט הולך ל-stderr |
| משימה נהרגת ב-OOM RAM (לא GPU) | הגדל `--mem=` ב-SBATCH |
| Timeout באמצע משימה | הגדל `--time=`; וודא QoS מאפשר את הזמן |
| `git pull` עם submodule נכשל | `git submodule deinit -f <name>` או טפל ב-`.gitmodules` |

---

## 11. טיפים מתקדמים

### Job dependencies (שרשור משימות)
```bash
JOB1=$(sbatch --parsable step1.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 step2.sh)
sbatch --dependency=afterok:$JOB2 step3.sh
```

### Resubmit על כשל
```bash
#SBATCH --requeue                  # מאפשר requeue אוטומטי
#SBATCH --signal=B:USR1@120        # שולח SIGUSR1 שתי דקות לפני timeout
```

### בדיקת GPU בזמן ריצה
```bash
# בסקריפט SLURM:
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 10 > gpu.log &
GPU_LOG_PID=$!
python my_script.py
kill $GPU_LOG_PID
```

### Profile עם time/memory
```bash
/usr/bin/time -v python my_script.py 2>&1 | tail -20
```

### Output buffering
```bash
export PYTHONUNBUFFERED=1                # Python: כתיבה ישירה (לא buffered)
stdbuf -oL -eL python my_script.py       # אם אי-אפשר לשנות env
```

---

## 12. הפניות חיצוניות

- [SLURM docs](https://slurm.schedmd.com/documentation.html)
- [SLURM cheat sheet](https://slurm.schedmd.com/pdfs/summary.pdf)
- [HuggingFace docs](https://huggingface.co/docs)
- שרת ה-IT של הטכניון (פנייה לבעיות גישה/QoS)
