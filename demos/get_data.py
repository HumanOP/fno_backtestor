import requests

url = "https://2centscapital-my.sharepoint.com/:u:/p/aniruddha_panda/EWGK6vj2oidKof0HTCEIF2MBxLSsKpR351Y7R4xcIPmzog?download=1"

output_path = "nifty_1min_desiquant.duckdb"

response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download completed successfully.")
else:
    print(f"Failed to download. Status code: {response.status_code}")