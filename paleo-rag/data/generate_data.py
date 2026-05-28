"""Synthetic data generator for Paleo RAG system."""

import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import config


EPOCHS = [
    "Кембрийский период (541-485 млн лет назад)",
    "Ордовикский период (485-444 млн лет назад)",
    "Силурийский период (444-419 млн лет назад)",
    "Девонский период (419-359 млн лет назад)",
    "Каменноугольный период (359-299 млн лет назад)",
    "Пермский период (299-252 млн лет назад)",
    "Триасовый период (252-201 млн лет назад)",
    "Юрский период (201-145 млн лет назад)",
    "Меловой период (145-66 млн лет назад)",
    "Палеогеновый период (66-23 млн лет назад)",
    "Неогеновый период (23-2.6 млн лет назад)",
    "Четвертичный период (2.6 млн лет назад - настоящее время)",
]

LOCATIONS = [
    "Россия, Сибирь",
    "Монголия, Гоби",
    "Китай, провинция Ляонин",
    "Аргентина, Патагония",
    "США, Монтана",
    "Канада, Альберта",
    "Танзания, формация Тендагуру",
    "Марокко, Атласские горы",
    "Германия, Золенхофен",
    "Великобритания, Дорсет",
    "Бразилия, формация Сантана",
    "Австралия, Квинсленд",
]

HALLS = [
    "Зал динозавров",
    "Зал древних рыб",
    "Зал аммонитов и моллюсков",
    "Зал первобытных млекопитающих",
    "Зал эволюции жизни",
    "Зал палеоботаники",
    "Интерактивный зал раскопок",
]

DIFFICULTIES = ["начальный", "средний", "продвинутый", "научный"]

EXHIBIT_TEMPLATES = [
    {
        "type": "exhibit",
        "title_template": "{dinosaur_name} ({epoch_short})",
        "content_template": "{dinosaur_name} — это род {diet} динозавров, живших в {epoch_full}. "
        "Длина взрослого экземпляра достигала {length} метров, вес — около {weight} тонн. "
        "Первые окаменелости были найдены в {location} в {year} году. "
        "Этот экспонат демонстрирует {feature}. Особый интерес представляет {detail}.",
    },
    {
        "type": "exhibit",
        "title_template": "Череп {animal_name} из {location_short}",
        "content_template": "Череп {animal_name}, обнаруженный в {location} в {year} году, "
        "представляет собой один из наиболее полных образцов вида. "
        "Размер черепа составляет {skull_size} см, что указывает на крупные размеры животного. "
        "Строение зубов свидетельствует о {diet_description}. "
        "На черепе видны следы {trace_type}, что даёт информацию о поведении или причинах смерти.",
    },
    {
        "type": "exhibit",
        "title_template": "Скелет {marine_animal} ({epoch_short})",
        "content_template": "{marine_animal} — морское пресмыкающееся эпохи {epoch_full}. "
        "Длина тела достигала {length} метров. Обитало в тёплых морях, питалось {food}. "
        "Экспонат найден в {location} в {year} году. Сохранность скелета составляет {preservation}%. "
        "Уникальная особенность — {unique_feature}.",
    },
    {
        "type": "exhibit",
        "title_template": "Отпечаток древнего растения: {plant_name}",
        "content_template": "{plant_name} — вымершее растение {period} периода. "
        "Отпечаток листа/ствола обнаружен в {location} в {year} году. "
        "Размер отпечатка: {size} см. По структуре листьев можно судить о {climate_info}. "
        "Растение играло важную роль в экосистеме того времени, служа пищей для {herbivores}.",
    },
    {
        "type": "exhibit",
        "title_template": "Аммонит рода {ammonite_genus}",
        "content_template": "Аммониты — вымершие головоногие моллюски, обитавшие в морях с {epoch_start} по {epoch_end}. "
        "Диаметр раковины этого экземпляра: {diameter} см. "
        "Место находки: {location}, {year} год. "
        "По аммонитам палеонтологи определяют возраст горных пород (биостратиграфия). "
        "Узор sutures (швов) на раковине характерен для рода {ammonite_genus}.",
    },
]

DATING_TEMPLATES = [
    {
        "type": "dating_method",
        "title_template": "Радиометрическое датирование: {method_name}",
        "content_template": "{method_name} — метод определения возраста горных пород и окаменелостей, "
        "основанный на распаде радиоактивных изотопов. "
        "Период полураспада: {half_life}. Применяется для образцов возрастом от {min_age} до {max_age}. "
        "Точность метода составляет ±{accuracy} млн лет. "
        "В музее этот метод использовался для датирования экспонатов из {location}.",
    },
    {
        "type": "dating_method",
        "title_template": "Биостратиграфическое датирование",
        "content_template": "Биостратиграфия — метод относительного датирования, основанный на анализе "
        "ископаемых организмов (индекс-фоссилий). "
        "Определённые виды существовали в ограниченный временной интервал, "
        "что позволяет коррелировать слои горных пород. "
        "В палеонтологии часто используются аммониты, трилобиты, фораминиферы. "
        "Метод не даёт абсолютного возраста, но позволяет установить последовательность событий.",
    },
    {
        "type": "dating_method",
        "title_template": "Уран-свинцовое датирование цирконов",
        "content_template": "Уран-свинцовый метод — один из самых точных способов датирования древних пород. "
        "Основан на распаде урана-238 в свинец-206 (период полураспада 4.47 млрд лет) "
        "и урана-235 в свинец-207 (период полураспада 704 млн лет). "
        "Применяется к минералу циркон, который устойчив к выветриванию. "
        "Позволяет датировать породы возрастом от 1 млн до 4.5 млрд лет с точностью до 0.1%.",
    },
]

FAQ_TEMPLATES = [
    {
        "type": "faq",
        "title_template": "Как определяют возраст динозавров?",
        "content_template": "Возраст динозавров определяют несколькими методами. "
        "Радиометрическое датирование измеряет распад радиоактивных изотопов в породах вокруг окаменелости. "
        "Биостратиграфия использует индекс-фоссилии для корреляции слоёв. "
        "Также применяют магнитостратиграфию (анализ магнитных свойств пород) и хемостратиграфию. "
        "Комбинация методов даёт наиболее точный результат.",
    },
    {
        "type": "faq",
        "title_template": "Почему динозавры вымерли?",
        "content_template": "Основная гипотеза — падение астероида диаметром около 10 км "
        "66 миллионов лет назад (граница мела и палеогена). "
        "Удар вызвал пожары, цунами и «ядерную зиму» из-за пыли в атмосфере. "
        "Дополнительные факторы: вулканическая активность (траппы Декана в Индии), "
        "изменение уровня моря и климата. Выжили только птицы — потомки тероподов.",
    },
    {
        "type": "faq",
        "title_template": "Как сохраняются окаменелости?",
        "content_template": "Окаменелости образуются при замещении органических тканей минералами (перминерализация). "
        "Необходимы условия: быстрое захоронение, отсутствие кислорода, наличие минерализованной воды. "
        "Типы сохранности: полная (янтарь, вечная мерзлота), отпечатки, ядра, casts (слепки). "
        "Мягкие ткани сохраняются крайне редко — известны случаи сохранения перьев, кожи, сосудов.",
    },
    {
        "type": "faq",
        "title_template": "Можно ли клонировать динозавра?",
        "content_template": "На текущий момент — нет. ДНК разрушается со временем, "
        "максимальный возраст прочитанной ДНК — около 1.5 млн лет (мамонт). "
        "Динозавры вымерли 66 млн лет назад, их ДНК не сохранилась. "
        "Даже теоретически клонирование невозможно без целой молекулы ДНК. "
        "Учёные работают над «обратной эволюцией» птиц, чтобы активировать древние гены (зубы, хвост).",
    },
    {
        "type": "faq",
        "title_template": "Какой динозавр был самым большим?",
        "content_template": "Самыми крупными считаются завроподы: Argentinosaurus, Patagotitan, Dreadnoughtus. "
        "Их длина достигала 35-40 метров, вес — 70-100 тонн. "
        "Для сравнения: синий вес (крупнейшее современное животное) весит до 150 тонн. "
        "Точный размер определить сложно, так как скелеты часто неполные. "
        "Новые находки постоянно уточняют наши представления о гигантах мезозоя.",
    },
    {
        "type": "faq",
        "title_template": "Что такое трансильванский дракула?",
        "content_template": "Это популярный миф. Динозавр Dracorex hogwartsia назван в честь Хогвартса из Гарри Поттера, "
        "а не графа Дракулы. Некоторые динозавры найдены в Трансильвании (Румыния), "
        "например, рудраптор (Balaur bondoc). Но связь с вампирами — лишь маркетинговый ход. "
        "Реальные хищные динозавры были гораздо страшнее любых легенд.",
    },
    {
        "type": "faq",
        "title_template": "Как работают палеонтологические раскопки?",
        "content_template": "Раскопки начинаются с разведки: геологи ищут выходы пород нужного возраста. "
        "Затем закладывают квадраты, снимают грунт послойно, документируют каждое находку. "
        "Кости очищают, укрепляют клеящими составами, оборачивают в гипсовые рубашки. "
        "В лаборатории образец готовят под микроскопом, делают КТ-сканы. "
        "Важно сохранить контекст находки — положение в слое говорит о многом.",
    },
    {
        "type": "faq",
        "title_template": "Правда ли, что некоторые динозавры имели перья?",
        "content_template": "Да, это доказанный факт. Перья найдены у многих теропод: "
        "Sinosauropteryx, Microraptor, Velociraptor, даже у предков T-Rex. "
        "Перья изначально служили для терморегуляции и демонстрации, позже — для полёта. "
        "Некоторые динозавры имели сложный окрас (меланосомы сохранились). "
        "Птицы — прямые потомки динозавров, так что технически динозавры не вымерли полностью.",
    },
]

DINOSAUR_NAMES = [
    "Tyrannosaurus rex", "Triceratops horridus", "Velociraptor mongoliensis",
    "Stegosaurus stenops", "Brachiosaurus altithorax", "Diplodocus longus",
    "Ankylosaurus magniventris", "Parasaurolophus walkeri", "Spinosaurus aegyptiacus",
    "Allosaurus fragilis", "Pteranodon longiceps", "Archaeopteryx lithographica",
    "Iguanodon bernissartensis", "Carnotaurus sastrei", "Therizinosaurus cheloniformis",
    "Pachycephalosaurus wyomingensis", "Baryonyx walkeri", "Oviraptor philoceratops",
    "Maiasaura peeblesorum", "Coelophysis bauri", "Plateosaurus engelhardti",
    "Herrerasaurus ischigualastensis", "Eoraptor lunensis", "Massospondylus carinatus",
]

MARINE_ANIMALS = [
    "Plesiosaurus dolichodeirus", "Ichthyosaurus communis", "Mosasaurus hoffmannii",
    "Elasmosaurus platyurus", "Liopleurodon ferox", "Shonisaurus popularis",
]

PLANT_NAMES = [
    "Glossopteris indica", "Lepidodendron aculeatum", "Calamites suckowii",
    "Neuropteris ovata", "Alethopteris grandini", "Sigillaria elegans",
]

AMMONITE_GENERA = [
    "Amaltheus", "Dactylioceras", "Harpoceras", "Lytoceras", "Phylloceras",
    "Stephanoceras", "Parkinsonia", "Kosmoceras", "Cardioceras", "Quenstedtoceras",
]

DIETS = ["травоядных", "хищных", "всеядных"]
DIET_DESCRIPTIONS = [
    "хищном питании (мясная диета)",
    "травоядном питании (растительная диета)",
    "всеядном питании (смешанная диета)",
]

FOODS = ["рыбой и кальмарами", "мелкими морскими рептилиями", "планктоном", "моллюсками"]

FEATURES = [
    "полный скелет в характерной позе",
    "череп с сохранившимися зубами",
    "отпечатки кожи и перьев",
    "кости с признаками патологий",
    "сочленённый скелет juveniles особи",
]

DETAILS = [
    "следы укусов других хищников",
    "медикаментозные изменения костей",
    "необычное оперение хвоста",
    "признаки внутривидовой агрессии",
    "адаптации к специфической диете",
]

TRACE_TYPES = [
    "заживших травм",
    "паразитарных поражений",
    "возрастных изменений",
    "патологического роста",
]

UNIQUE_FEATURES = [
    "сохранение хрящевой ткани",
    "трёхмерная структура костей",
    "остатки желудочного содержимого",
    "эмбрионы внутри скелета",
]

CLIMATE_INFOS = [
    "влажном тропическом климате",
    "сезонно-аридном климате",
    "умеренном влажном климате",
    "прибрежном морском климате",
]

HERBIVORES = ["динозавров-завропод", "орнитопод", "протоцератопсов", "ранних млекопитающих"]

METHOD_NAMES = [
    "Калий-аргоновый метод (K-Ar)",
    "Аргон-аргоновый метод (Ar-Ar)",
    "Рубидий-стронциевый метод (Rb-Sr)",
    "Самарий-неодимовый метод (Sm-Nd)",
    "Уран-ториевый метод (U-Th)",
]

HALF_LIFES = [
    "1.25 млрд лет",
    "1.42 млрд лет",
    "49 млрд лет",
    "106 млрд лет",
    "75 тыс. лет",
]

MIN_AGES = ["100 тыс.", "1 млн", "10 млн", "100 млн"]
MAX_AGES = ["100 млн", "1 млрд", "4.5 млрд", "10 млрд"]
ACCURACIES = ["0.1", "0.5", "1.0", "2.0", "5.0"]

PRESERVATIONS = ["65", "72", "78", "85", "91", "94"]
SKULL_SIZES = ["45", "62", "78", "95", "120", "145"]
LENGTHS = ["3", "6", "9", "12", "18", "25", "35"]
WEIGHTS = ["0.5", "1.2", "3", "8", "25", "50", "80"]
YEARS = ["1822", "1842", "1861", "1877", "1898", "1905", "1923", "1964", "1971", "1986", "1993", "2001", "2008", "2014", "2019"]
DIAMETERS = ["8", "12", "18", "25", "34", "42", "56"]
SIZES = ["5×3", "12×8", "25×15", "45×30", "80×50"]

LOCATION_SHORTS = ["Сибири", "Монголии", "Китая", "Аргентины", "США", "Канады", "Танзании", "Марокко", "Германии", "Великобритании"]


def generate_exhibit_doc(doc_id: int, template: dict[str, Any]) -> dict[str, Any]:
    """Generate a single exhibit document from template."""
    epoch = random.choice(EPOCHS)
    epoch_short = epoch.split(" (")[0]
    location = random.choice(LOCATIONS)
    location_short = random.choice(LOCATION_SHORTS)
    
    if "dinosaur_name" in template["title_template"]:
        title = template["title_template"].format(
            dinosaur_name=random.choice(DINOSAUR_NAMES),
            epoch_short=epoch_short,
        )
        content = template["content_template"].format(
            dinosaur_name=title.split(" (")[0],
            diet=random.choice(DIETS),
            epoch_full=epoch,
            epoch_short=epoch_short,
            length=random.choice(LENGTHS),
            weight=random.choice(WEIGHTS),
            location=location,
            year=random.choice(YEARS),
            feature=random.choice(FEATURES),
            detail=random.choice(DETAILS),
        )
    elif "Череп" in template["title_template"]:
        animal = random.choice(DINOSAUR_NAMES + MARINE_ANIMALS)
        title = template["title_template"].format(
            animal_name=animal,
            location_short=location_short,
        )
        content = template["content_template"].format(
            animal_name=animal,
            location=location,
            year=random.choice(YEARS),
            skull_size=random.choice(SKULL_SIZES),
            diet_description=random.choice(DIET_DESCRIPTIONS),
            trace_type=random.choice(TRACE_TYPES),
        )
    elif "Скелет" in template["title_template"]:
        marine = random.choice(MARINE_ANIMALS)
        title = template["title_template"].format(
            marine_animal=marine,
            epoch_short=epoch_short,
        )
        content = template["content_template"].format(
            marine_animal=marine,
            epoch_full=epoch,
            length=random.choice(LENGTHS),
            food=random.choice(FOODS),
            location=location,
            year=random.choice(YEARS),
            preservation=random.choice(PRESERVATIONS),
            unique_feature=random.choice(UNIQUE_FEATURES),
        )
    elif "растения" in template["title_template"]:
        plant = random.choice(PLANT_NAMES)
        period = random.choice(["Кембрийского", "Ордовикского", "Силурийского", "Девонского", 
                               "Каменноугольного", "Пермского", "Триасового", "Юрского", "Мелового"])
        title = template["title_template"].format(plant_name=plant)
        content = template["content_template"].format(
            plant_name=plant,
            period=period,
            location=location,
            year=random.choice(YEARS),
            size=random.choice(SIZES),
            climate_info=random.choice(CLIMATE_INFOS),
            herbivores=random.choice(HERBIVORES),
        )
    else:
        ammonite = random.choice(AMMONITE_GENERA)
        epoch_start = random.randint(541, 200)
        epoch_end = epoch_start - random.randint(10, 100)
        title = template["title_template"].format(ammonite_genus=ammonite)
        content = template["content_template"].format(
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            diameter=random.choice(DIAMETERS),
            location=location,
            year=random.choice(YEARS),
            ammonite_genus=ammonite,
        )
    
    return {
        "id": doc_id,
        "type": template["type"],
        "title": title,
        "content": content,
        "metadata": {
            "epoch": epoch_short,
            "location": location,
            "hall": random.choice(HALLS),
            "difficulty": random.choice(DIFFICULTIES),
        }
    }


def generate_dating_doc(doc_id: int, template: dict[str, Any]) -> dict[str, Any]:
    """Generate a dating method document from template."""
    location = random.choice(LOCATIONS)
    
    if "method_name" in template["title_template"]:
        title = template["title_template"].format(method_name=random.choice(METHOD_NAMES))
        content = template["content_template"].format(
            method_name=title.split(": ")[1],
            half_life=random.choice(HALF_LIFES),
            min_age=random.choice(MIN_AGES),
            max_age=random.choice(MAX_AGES),
            accuracy=random.choice(ACCURACIES),
            location=location,
        )
    else:
        title = template["title_template"]
        content = template["content_template"]
    
    return {
        "id": doc_id,
        "type": template["type"],
        "title": title,
        "content": content,
        "metadata": {
            "epoch": "N/A",
            "location": location,
            "hall": "Зал палеоботаники",
            "difficulty": "продвинутый",
        }
    }


def generate_faq_doc(doc_id: int, template: dict[str, Any]) -> dict[str, Any]:
    """Generate an FAQ document from template."""
    title = template["title_template"]
    content = template["content_template"]
    
    return {
        "id": doc_id,
        "type": template["type"],
        "title": title,
        "content": content,
        "metadata": {
            "epoch": "N/A",
            "location": "N/A",
            "hall": "Интерактивный зал раскопок",
            "difficulty": random.choice(["начальный", "средний"]),
        }
    }


def generate_all_documents(num_docs: int = None, seed: int = None) -> list[dict[str, Any]]:
    """Generate all synthetic documents for the paleontology museum."""
    num_docs = num_docs or config.NUM_DOCUMENTS
    seed = seed if seed is not None else config.RANDOM_SEED
    
    random.seed(seed)
    
    all_docs = []
    doc_id = 0
    
    templates_by_type = {
        "exhibit": EXHIBIT_TEMPLATES,
        "dating_method": DATING_TEMPLATES,
        "faq": FAQ_TEMPLATES,
    }
    
    weights = [0.6, 0.2, 0.2]
    
    progress_bar = tqdm(total=num_docs, desc="Generating documents")
    
    while doc_id < num_docs:
        template_type = random.choices(
            list(templates_by_type.keys()),
            weights=weights,
            k=1
        )[0]
        
        template = random.choice(templates_by_type[template_type])
        
        if template_type == "exhibit":
            doc = generate_exhibit_doc(doc_id, template)
        elif template_type == "dating_method":
            doc = generate_dating_doc(doc_id, template)
        else:
            doc = generate_faq_doc(doc_id, template)
        
        all_docs.append(doc)
        doc_id += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    return all_docs


def save_documents(docs: list[dict[str, Any]], output_path: Path) -> None:
    """Save documents to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")




def load_documents(input_path: Path) -> list[dict[str, Any]]:
    """Load documents from JSONL file."""
    documents = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
    
    return documents
def main():
    """Main entry point for data generation."""
    print(f"Generating {config.NUM_DOCUMENTS} synthetic documents...")
    
    docs = generate_all_documents()
    
    save_documents(docs, config.RAW_DATA_FILE)
    
    print(f"Documents saved to {config.RAW_DATA_FILE}")


if __name__ == "__main__":
    main()
