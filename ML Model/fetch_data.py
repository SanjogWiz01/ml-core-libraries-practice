import requests
import json
import os

def fetch_nepali_assets():
    """Fetch Nepali asset data from public websites/APIs"""
    assets = []
    
    # Try Nepal Government open data or asset-related APIs
    urls = [
        "https://api.data.gov.np/v1/assets?limit=50",
        "https://api.wikidata.org/w/api.php?action=query&list=search&srsearch=nepali%20asset&format=json",
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "assets" in data:
                    assets.extend(data["assets"])
                elif "search" in data:
                    for item in data["search"]:
                        assets.append({
                            "title": item.get("title", ""),
                            "description": item.get("snippet", ""),
                            "category": "general"
                        })
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
    
    # If no external data fetched, use curated Nepali asset dataset
if not assets:
        nepali_assets = [
            {"id": 1, "name": "Pashupatinath Temple", "category": "religious", "base_desc": "Hindu temple complex on the banks of the Bagmati River in Kathmandu"},
            {"id": 2, "name": "Swayambhunath Stupa", "category": "religious", "base_desc": "Ancient Buddhist stupa known as the Monkey Temple on a hill in Kathmandu"},
            {"id": 3, "name": "Boudhanath Stupa", "category": "religious", "base_desc": "One of the largest spherical stupas in Nepal, important center of Tibetan Buddhism"},
            {"id": 4, "name": "Kathmandu Durbar Square", "category": "historical", "base_desc": "Historic plaza with palaces, temples, and courtyards in the heart of Kathmandu"},
            {"id": 5, "name": "Bhaktapur Durbar Square", "category": "historical", "base_desc": "Medieval palace complex with 55 windows and artistic courtyards"},
            {"id": 6, "name": "Patan Durbar Square", "category": "historical", "base_desc": "Ancient Newar palace complex with exquisite stone and metal carvings"},
            {"id": 7, "name": "Chitwan National Park", "category": "natural", "base_desc": "UNESCO World Heritage site with Bengal tigers and one-horned rhinos"},
            {"id": 8, "name": "Sagarmatha National Park", "category": "natural", "base_desc": "Home to Mount Everest and Himalayan wildlife"},
            {"id": 9, "name": "Lumbini", "category": "religious", "base_desc": "Birthplace of Lord Gautama Buddha"},
            {"id": 10, "name": "Mount Everest", "category": "natural", "base_desc": "World's highest peak at 8,848.86 meters"},
            {"id": 11, "name": "Pokhara Lake", "category": "natural", "base_desc": "Famous lake city with Phewa Tal reflecting the Annapurna range"},
            {"id": 12, "name": "Changunarayan Temple", "category": "religious", "base_desc": "Oldest Hindu temple with exquisite stone inscriptions near Bhaktapur"},
            {"id": 13, "name": "Sundarijal", "category": "natural", "base_desc": "Scenic watershed and forest area on the edge of Kathmandu Valley"},
            {"id": 14, "name": "Namche Bazaar", "category": "historical", "base_desc": "Famous Sherpa trading town gateway to Everest region"},
            {"id": 15, "name": "Muktinath", "category": "religious", "base_desc": "Sacred pilgrimage site for both Hindus and Buddhists at 3,800m altitude"},
        ]
        assets = nepali_assets
    
    return assets

if __name__ == "__main__":
    assets = fetch_nepali_assets()
    os.makedirs("data", exist_ok=True)
    with open("data/nepali_assets.json", "w", encoding="utf-8") as f:
        json.dump(assets, f, ensure_ascii=False, indent=2)
    print(f"Fetched {len(assets)} Nepali assets")
    for a in assets:
        print(f"  {a['id']}. {a['name']} - {a['category']}")