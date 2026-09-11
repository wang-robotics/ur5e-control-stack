import pathlib

p = pathlib.Path("outputs/train_vla_log.txt")
raw = p.read_bytes()
for enc in ("utf-8", "utf-16", "gbk"):
    try:
        txt = raw.decode(enc)
    except (UnicodeDecodeError, UnicodeError):
        continue
    lines = txt.splitlines()
    print(f"decoded as {enc}, {len(lines)} lines")
    if lines:
        print("first:", lines[0])
        print("last :", lines[-1])
    break
