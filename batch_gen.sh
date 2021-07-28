cat captions.txt | xargs -n 1 -P 0 -I {} ./cgd_venv/bin/python hq.py "{}";
