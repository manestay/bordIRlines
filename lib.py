import os

import datasets


def load_borderlines_hf(dataset_dir=None):
    if not dataset_dir:
        print("loading from the datasets hub...", end=" ")

        territories = datasets.load_dataset("manestay/borderlines", "territories")["train"]
        countries = datasets.load_dataset("manestay/borderlines", "countries")["train"]
        queries = datasets.load_dataset("manestay/borderlines", "queries")
    else:
        print(f"loading from {dataset_dir}...", end=" ")
        territories = datasets.load_from_disk(os.path.join(dataset_dir, "territories"))
        countries = datasets.load_from_disk(os.path.join(dataset_dir, "countries"))
        queries = datasets.load_from_disk(os.path.join(dataset_dir, "queries"))
    print("done")
    return territories, countries, queries


def get_territory_langs_map(
    territories_ds: datasets.Dataset, countries_info: dict[str, dict]
) -> dict[str, set[str]]:
    territory_langs_map = {
        t["Territory"]: set([countries_info[x]["Lang_Code"] for x in t["Claimants"]])
        for t in territories_ds
    }
    return territory_langs_map


def get_territory_claimants_and_langs_map(
    territories_ds: datasets.Dataset, countries_info: dict[str, dict]
) -> dict[str, dict]:
    territory_claimants_map = dict()
    for t in territories_ds:
        claimants = t["Claimants"]
        claimant_langs = [countries_info[x]["Lang_Code"] for x in claimants]
        territory_claimants_map[t["Territory"]] = {
            "claimants": claimants,
            "claimant_langs": claimant_langs,
        }

    return territory_claimants_map


def split_qid(qid):
    arr = qid.rsplit("_", 1)
    return arr[0].replace("_", " "), arr[1]


def join_into_qid(territory, lang):
    return f"{territory.replace(' ', '_')}_{lang}"


def get_territory_names(query_ds):
    return [split_qid(qid)[0] for qid in query_ds["QueryID"]]


def check_languages(languages, queries_ds=None, add_control=True):
    if not queries_ds:
        return languages

    all_languages = list(queries_ds.keys())
    if add_control:
        all_languages = ["control"] + all_languages

    if "all" in languages:
        print(f"using all languages: {all_languages}")
        return all_languages
    languages_ = []

    for lang in languages:
        if lang not in all_languages:
            print(f"lang {lang} not found, skipping...")
            continue
        languages_.append(lang)
    return languages_


# only fix territories which have a page in English
territory_fix_d = {
    "Jammu and Kashmir": "Jammu and Kashmir (union territory)",
    "Qaruh": "Qaruh Island",
    "Lunchinda-Pweto Province": "Luapula Province border dispute",
    "Dragonja River": "Dragonja",
    "Susta River dispute": "Susta territory",
    "Trans-Karakoram Tracts": "Trans-Karakoram Tract",
    "Doi Lang": "Doi Pha Hom Pok National Park",
    "Khao Phra Wihan": "Khao Phra Wihan National Park",
    "Fasht ad Dibal": "Fasht Dibal conflict",
    "Junagadh and Manavadar": "Junagadh State",
    "Mont Blanc summit dispute": "Mont Blanc",
    "South Kuril/Chishima Islands": "Kuril Islands dispute",
    "Bukit Jeli": "Malaysia–Thailand border",
    "Bhutanese exclaves": "History of Bhutan",
    "Umm al Maradim": "Umm al Maradim Island",
    "Matsu": "Matsu Islands",
    "Kalapani": "Kalapani territory",
    "Qasr": "Qasr, Lebanon",
    "Green Line": "Green Line (Israel)",
    "Palestine": "State of Palestine",
    "Sang": "Sang, Uttarakhand",
    "Pedra Branca": "Pedra Branca dispute",
    "Karki": "Karki, Azerbaijan",
    "Tashigang": "Tashigang, Himachal Pradesh",
    "Point 20": "Malaysia–Singapore border",
    "Marouini River": "Maroni (river)",
    "Georgia": "Georgia (country)",
    "David Gareja monastery complex": "David Gareji monastery complex",
    "Doumeira Island": "Doumeira Islands",
    "Hala'ib Triangle": "Halaib Triangle",
    "Socotra Archipelago": "Socotra Governorate",
    "Guayana Esequiba": "Guyana–Venezuela territorial dispute",
    "Arroyo de la Invernada": "Masoller",
    "Rincón de Artigas": "Masoller",
    "Vila Albornoz": "Vila Thomaz Albornoz",
    "Isla Brasilera": "Brazilian Island",
    "Ilha Brasileira": "Brazilian Island",
    "Gegharkunik province": "Gegharkunik Province",
    "Nakhichevan Autonomous Republic": "Nakhchivan Autonomous Republic",
    "Qazakh Rayon": "Qazax District",
    "Chishima Islands": "Kuril Islands",
    "Kuril": "Kuril Islands",
    "Dokdo": "Liancourt Rocks",
    "Takeshima": "Liancourt Rocks",
    "Mekong river": "Mekong",
    "Stung Treng Province": "Stung Treng province",
    "Saltoro Ridge": "Saltoro Mountains",
    "Ukatnyy": "Ukatny Island",
    "Zhestky": "Ukatny Island",
    "Malyy Zhemchuzhnyy": "Ukatny Island",
    "Baekdu Mountain": "Paektu Mountain",
    "Jadhang": "Sang, Uttarakhand",
    "Heixiazi": "Bolshoy Ussuriysky Island",
    "Shaksgam Valley": "Shaksgam River",
}


# territories with an English-speaking claimant
EN_TERRS = set(
    [
        "Abyei_en",
        "Heglig_en",
        "Kafia_Kingi_en",
        "Chagos_Archipelago_en",
        "Ilemi_Triangle_en",
        "KaNgwane_en",
        "Ingwavuma_en",
        "Logoba_en",
        "Moyo_District_en",
        "Chiengi_en",
        "Lunchinda-Pweto_Province_en",
        "Lunkinda_River_en",
        "Pweto_en",
        "Mbamba_Bay_en",
        "Lake_Nyasa_en",
        "Okpara_River_en",
        "Sindabezi_Island_en",
        "Bajo_Nuevo_Bank_en",
        "Navassa_Island_en",
        "Sapodilla_Cayes_en",
        "Serranilla_Bank_en",
        "Guayana_Esequiba_en",
        "Essequibo_River_en",
        "Ankoko_Island/Isla_de_Anacoco_en",
        "Falkland_Islands_en",
        "South_Georgia_and_the_South_Sandwich_Islands_en",
        "Tigri_Area_en",
        "Courantyne_River_en",
        "Ashmore_and_Cartier_Islands_en",
        "Pedra_Branca_en",
        "Singapore_Strait_en",
        "Point_20_en",
        "Gibraltar_en",
        "Rockall_en",
        "Matthew_Island_and_Hunter_Island_en",
        "Minerva_Reefs_en",
        "Swains_Island_en",
        "Wake_Island_en",
    ]
)


ALL_LANGS = [
    "sl",
    "ur",
    "sw",
    "uz",
    "vi",
    "sq",
    "ms",
    "km",
    "hy",
    "da",
    "ky",
    "mg",
    "mn",
    "ja",
    "el",
    "it",
    "is",
    "ru",
    "tl",
    "so",
    "pt",
    "uk",
    "sr",
    "sn",
    "ht",
    "bs",
    "my",
    "ar",
    "hr",
    "nl",
    "bn",
    "ne",
    "hi",
    "ka",
    "az",
    "ko",
    "id",
    "fr",
    "es",
    "en",
    "fa",
    "lo",
    "iw",
    "th",
    "tr",
    "zht",
    "zhs",
    "ti",
    "tg",
]
