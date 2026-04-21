import requests
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    FEWSNET_BASE_URL, HDX_HAPI_BASE_URL,
    COUNTRY_CODE, DATA_RAW,
    FEWSNET_TOKEN, HDX_APP_IDENTIFIER
)

def _get(url, params, label):
    headers = {}
    if FEWSNET_TOKEN and "fdw.fews.net" in url:
        headers["Authorization"] = f"Token {FEWSNET_TOKEN}"
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"  [!] HTTP {resp.status_code} on {label} (attempt {attempt}/3): {e}")
        except requests.exceptions.RequestException as e:
            print(f"  [!] Request failed on {label} (attempt {attempt}/3): {e}")
        if attempt < 3:
            time.sleep(2)
    return None

def _get_all_pages(base_url, params, label):
    results = []
    url = base_url
    page = 1
    while url:
        print(f"  -> page {page} ...", end=" ", flush=True)
        data = _get(url, params if page == 1 else {}, label)
        if data is None:
            break
        if isinstance(data, list):
            results.extend(data)
            break
        else:
            batch = data.get("results", [])
            results.extend(batch)
            url = data.get("next")
            page += 1
            if not batch:
                break
    print(f"total: {len(results)} records")
    return results

def _save(data, filename):
    os.makedirs(DATA_RAW, exist_ok=True)
    path = os.path.join(DATA_RAW, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [ok] Saved -> {path}")
    return path

def fetch_fewsnet_classifications():
    print("\n[1/3] FEWS NET -- IPC classifications (livelihood zones)...")
    url = f"{FEWSNET_BASE_URL}/ipcclassification/"
    params = {"country": "MG", "format": "json", "page_size": 200}
    return _get_all_pages(url, params, "FEWS classifications")

def fetch_fewsnet_population():
    print("\n[2/3] FEWS NET -- IPC population estimates...")
    url = f"{FEWSNET_BASE_URL}/ipcpopulation/"
    params = {"country": "MG", "format": "json", "page_size": 200}
    return _get_all_pages(url, params, "FEWS population")

def fetch_hdx_food_security():
    print("\n[3/3] HDX HAPI -- food security phases...")
    if not HDX_APP_IDENTIFIER:
        print("  [!] HDX_APP_IDENTIFIER not set in .env -- skipping")
        return []
    url = f"{HDX_HAPI_BASE_URL}/food-security-nutrition-poverty/food-security"
    params = {
        "location_code": COUNTRY_CODE,
        "output_format": "json",
        "limit": 1000,
        "app_identifier": HDX_APP_IDENTIFIER,
    }
    data = _get(url, params, "HDX food security")
    if data is None:
        return []
    results = data.get("data", data.get("results", []))
    print(f"  total: {len(results)} records")
    return results

def run_ingestion():
    print("\n=== Data Ingestion: Madagascar Food Security ===")
    print(f"    Timestamp: {datetime.utcnow().isoformat()}Z")

    classifications = fetch_fewsnet_classifications()
    if classifications:
        _save(classifications, "fewsnet_classifications_mdg.json")

    population = fetch_fewsnet_population()
    if population:
        _save(population, "fewsnet_population_mdg.json")

    hdx = fetch_hdx_food_security()
    if hdx:
        _save(hdx, "hdx_food_security_mdg.json")

    print("\n=== Ingestion complete ===")
    print(f"  classifications : {len(classifications)} records")
    print(f"  population      : {len(population)} records")
    print(f"  hdx             : {len(hdx)} records")

if __name__ == "__main__":
    run_ingestion()
