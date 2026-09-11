"""Download UCI Online Retail dataset."""
import requests
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_retail.xlsx")
out_path = os.path.abspath(out_path)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"Downloading from {url} ...")
resp = requests.get(url, stream=True, timeout=300)
resp.raise_for_status()

total = int(resp.headers.get("content-length", 0))
downloaded = 0
with open(out_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\rProgress: {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\rDownloaded: {downloaded/1024/1024:.1f} MB", end="", flush=True)

print(f"\nSaved to {out_path}")
print(f"File size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
