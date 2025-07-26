import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import requests
from opencc import OpenCC

from lib import load_borderlines_hf, territory_fix_d
from wiki_lib import parse_and_clean_wikicode

BASE_URL = "https://{lang}.wikipedia.org/w/api.php"
OUT_DIR = Path("data/raw/wikipedia")
TIMESTAMP = "2023-05-15"
# Note: Wikipedia only has 1 chinese version
WIKI_LANG = {"zhs": "zh", "zht": "zh", "iw": "he"}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", "-d", type=Path, default=None)
parser.add_argument("--out_dir", "-o", type=Path, default=OUT_DIR)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--multilingual", "-m", action="store_true")


@lru_cache(maxsize=1000)
def get_wikipedia_langlinks(title, timestamp=None, langs=[]):
    # Retrieve page IDs and interlanguage links
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "langlinks",
        "lllimit": "max",
        "redirects": 1,
    }

    if timestamp:
        params.update({"rvstart": timestamp + "T23:59:59Z", "rvdir": "older"})

    response = requests.get(url=BASE_URL.format(lang="en"), params=params)
    data = response.json()
    pages = list(data["query"]["pages"].values())

    # Check if page exists
    if "missing" in pages[0] or "langlinks" not in pages[0]:
        print(f"WARNING: no langlinks for {title}")
        return []

    # Get interlanguage link
    langlinks = {x["lang"]: x["*"] for x in pages[0]["langlinks"]}
    if langs:
        langlinks_ = {
            lang: langlinks.get(WIKI_LANG.get(lang, lang), f"None{title}") for lang in langs
        }
        langlinks = langlinks_
    return list(langlinks.items())


def get_wikipedia_article(title, timestamp=None, lang="en", filter_short=True):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content|timestamp",
        "rvslots": "main",
        "rvlimit": 1,
    }

    if timestamp:
        params.update({"rvstart": timestamp + "T23:59:59Z", "rvdir": "older"})

    # handle specific codes
    convert_fn = cc_t2s.convert if lang == "zhs" else cc_s2t.convert if lang == "zht" else None

    lang = WIKI_LANG.get(lang, lang)
    wiki_url = BASE_URL.format(lang=lang)

    response = requests.get(url=wiki_url, params=params)
    data = response.json()
    pages = list(data["query"]["pages"].values())

    for page in pages:
        if "missing" in page:
            return {}
        elif timestamp and "revisions" not in page:
            # exists, but after the timestamp. Get the oldest version of the article
            params["rvdir"] = "newer"
            response = requests.get(url=wiki_url, params=params)
            data = response.json()
            page = list(data["query"]["pages"].values())[0]

            timestamp = page["revisions"][0]["timestamp"]
            print(f"WARNING: used {timestamp} revision for {title}")

        page_id = list(data["query"]["pages"].keys())[0]

        content = page["revisions"][0]["slots"]["main"]["*"]
        timestamp = page["revisions"][0]["timestamp"]

        json_obj = {
            "title": title if not convert_fn else convert_fn(title),
            "timestamp": timestamp,
            "wiki_id": page_id,
        }

        json_obj["article"] = parse_and_clean_wikicode(content, lang)
        if filter_short:
            json_obj["article"] = [x for x in json_obj["article"] if len(x) > 20]
        if convert_fn:
            json_obj["article"] = [convert_fn(x) for x in json_obj["article"]]
    return json_obj


def get_title2id(entity2ids, add_not_found=True):
    title2id = {}
    if not entity2ids:
        return title2id

    for x in entity2ids.values():
        for wiki_id, title in zip(x.ids, x.titles):
            if title in title2id:
                continue
            title2id[title] = wiki_id
        if add_not_found:
            for title in x.not_found:
                title2id[title] = None
    return title2id


class TerritoryWikiMeta:
    def __init__(self, ids=None, titles=None, not_found=None):
        self.ids = ids or []
        self.titles = titles or []
        self.not_found = set(not_found or [])

    def to_dict(self, all_list=False):
        not_found = self.not_found if not all_list else list(self.not_found)
        return dict(ids=self.ids, titles=self.titles, not_found=not_found)

    @staticmethod
    def from_terr_dict(terr_d):
        terr2twm = defaultdict(TerritoryWikiMeta)
        for terr, d in terr_d.items():
            twm = TerritoryWikiMeta.from_dict(d)
            terr2twm[terr] = twm

        return terr2twm

    @staticmethod
    def from_dict(d):
        twm = TerritoryWikiMeta(d["ids"], d["titles"], d["not_found"])

        return twm

    def __repr__(self):
        return repr(self.to_dict())


def get_terr2twm():
    return defaultdict(TerritoryWikiMeta)


if __name__ == "__main__":
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    ## handle metadata
    metadata_dir = args.out_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    langs_title2id = defaultdict(dict)
    langs_entity2ids = defaultdict(get_terr2twm)
    if not args.overwrite:
        for entity2ids_path in metadata_dir.glob("entity2ids.*.json"):
            lang = entity2ids_path.suffixes[0][1:]
            with open(entity2ids_path) as f:
                terr2info = json.load(f)
                twms = TerritoryWikiMeta.from_terr_dict(terr2info)
                langs_entity2ids[lang] = twms

            langs_title2id[lang] = get_title2id(langs_entity2ids[lang])
    ##

    # simplified & traditional Chinese converters
    cc_s2t = OpenCC("s2t")
    cc_t2s = OpenCC("t2s")

    territories, countries, queries = load_borderlines_hf(args.dataset_dir)
    countries_info = {x["Country"]: x for x in countries}
    i = 0
    try:  # TODO: remove after bugs fixed
        for row in territories:
            territory_name = row["Territory"]
            claimants = row["Claimants"]
            print(f"processing {territory_name}")
            titles = territory_name.split("/")
            titles.extend(claimants)
            titles = [territory_fix_d.get(x, x) for x in titles]
            titles = [("en", x) for x in titles]

            if args.multilingual:
                langs = set([countries_info[x]["Lang_Code"] for x in claimants])
                langs.discard("en")
                if langs:
                    titles_multi = []
                    for _, title in titles:
                        # TODO: update to only call API if title in lang does not exist
                        links = get_wikipedia_langlinks(title, TIMESTAMP, langs=frozenset(langs))
                        titles_multi.extend(links)
                    titles.extend(titles_multi)

            for lang, title in titles:
                entry = langs_entity2ids[lang][territory_name]
                title2id = langs_title2id[lang]
                if title.startswith("None"):
                    entry.not_found.add(title[4:])
                    continue

                if title in title2id:
                    if title not in entry.titles and title2id[title]:
                        entry.titles.append(title)
                        entry.ids.append(title2id[title])
                    print(f"  {title}, {lang} seen")
                    continue

                article = get_wikipedia_article(title, TIMESTAMP, lang)

                if not article:
                    print(f"no Wikipedia page found for {title}, skipping")
                    entry.not_found.add(title)
                    continue

                lang_dir = args.out_dir / lang
                lang_dir.mkdir(exist_ok=True, parents=True)
                wiki_id = article["wiki_id"]
                out_name = lang_dir / f"Q{wiki_id}.json"

                with out_name.open("w") as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)
                entry.titles.append(title)
                entry.ids.append(wiki_id)
                title2id[title] = wiki_id
            i += 1
    finally:
        for lang, twms in langs_entity2ids.items():
            entity2ids_path = metadata_dir / f"entity2ids.{lang}.json"

            with entity2ids_path.open("w") as f:
                json.dump(
                    {k: v.to_dict(all_list=True) for k, v in twms.items()},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            all_not_found = [x.not_found for x in twms.values()]
            not_found_set = set([x for subl in all_not_found for x in subl])
            if len(not_found_set):
                print(f"for {lang}, {len(not_found_set)} were not found: ")
                print(not_found_set)
