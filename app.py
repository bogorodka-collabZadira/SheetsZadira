# app.py
from flask import Flask, request, jsonify
import pandas as pd
from natasha import Segmenter, NewsNERTagger, NewsEmbedding, Doc
import re
import os

app = Flask(__name__)

# === МУСОРНЫЕ СЛОВА ===
MUSOR = {
    "там", "нет", "зачем", "здравствуйте", "фото", "вы", "день", "ночь", "утро", "вечер",
    "здесь", "благодарю", "зафиксировала", "господи", "антоновкой", "позвать", "горячей",
    "линии", "обращение", "оформил", "оформила", "изображение", "мне", "вранье", "скидка",
    "везде", "картах", "куратор", "ооочень", "вам", "вообще", "полбагодарила", "ui", "ux",
    "подскажите", "пожалуйста", "спасибо", "хорошего", "всего", "добрый", "доброе", "привет",
    "увы", "но", "поняла", "еще", "сообщила", "на гд", "на гл", "вот", "ага", "ох", "ах",
    "ваша", "поддержка", "оператор"
}
MUSOR = {w.lower() for w in MUSOR}

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def is_word_junk(word: str) -> bool:
    w_clean = word.strip('.').strip()
    if len(w_clean) == 1:
        if re.fullmatch(r'[А-ЯЁ]', w_clean):
            return False
        else:
            return True
    w = w_clean.lower()
    if w in MUSOR:
        return True
    if re.search(r'[0-9]', w):
        return True
    if not re.fullmatch(r'[а-яё]+', w):
        return True
    return False

def is_valid_fio_phrase(phrase: str) -> bool:
    words = phrase.split()
    if len(words) < 2 or len(words) > 3:
        return False
    return not any(is_word_junk(w) for w in words)

def deduplicate_fio_variants(fio_list):
    """Оставляет только самую полную форму (3 компонента), удаляя 2-компонентные, если есть полная с теми же именем/отчеством."""
    if not fio_list:
        return []
    full_fios = []
    partial_fios = []
    for fio in fio_list:
        parts = fio.split()
        if len(parts) == 3:
            full_fios.append(fio)
        elif len(parts) == 2:
            partial_fios.append(fio)
    full_name_triples = []
    for fio in full_fios:
        parts = fio.split()
        full_name_triples.append((parts[0], parts[1], parts[2]))
    kept = set(full_fios)
    for fio in partial_fios:
        parts = fio.split()
        keep = True
        for fam, name, patr in full_name_triples:
            if len(parts) == 2 and parts[0] == name and parts[1] == patr:
                keep = False
                break
            if len(parts) == 2 and parts[0] == fam and parts[1] == name:
                keep = False
                break
        if keep:
            kept.add(fio)
    result = []
    seen = set()
    for fio in fio_list:
        if fio in kept and fio not in seen:
            result.append(fio)
            seen.add(fio)
    return result

# === ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ ===
def process_data(df_input):
    # Инициализация Natasha
    segmenter = Segmenter()
    ner_tagger = NewsNERTagger(NewsEmbedding())

    # Регулярка для ФИО
    fio_pattern = re.compile(
        r'\b'
        r'[А-ЯЁ][а-яё]+'
        r'(?:\s+[А-ЯЁ][а-яё]*\.?\s*[А-ЯЁ][а-яё]*\.?|(?:\s+[А-ЯЁ][а-яё]+){1,2})'
        r'\b'
    )

    results = []
    for text in df_input["text"]:
        raw = str(text)
        candidates = set()

        # --- NER ---
        doc = Doc(raw)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        for span in doc.spans:
            if span.type == "PER" and is_valid_fio_phrase(span.text):
                candidates.add(span.text)

        # --- Regex ---
        matches = fio_pattern.findall(raw)
        for match in matches:
            if is_valid_fio_phrase(match):
                candidates.add(match)

        # --- Чистка дублей и неполных форм ---
        candidate_list = list(candidates)
        cleaned_fio = deduplicate_fio_variants(candidate_list)

        # --- Резерв: номер 4*** ---
        result_str = ""
        if cleaned_fio:
            result_str = "\n".join(cleaned_fio)
        else:
            numbers = re.findall(r'\b4\d{3}\b', raw)
            if numbers:
                result_str = "\n".join(numbers)

        results.append({
            "Обращение": raw,
            "ФИО или Номер": result_str
        })

    # === ДОПОЛНИТЕЛЬНАЯ ОБРАБОТКА: УЛУЧШЕННЫЙ ПОИСК И РАЗНОС ПО СТРОКАМ ===
    temp_df = pd.DataFrame(results)

    # 1) Дополнительный поиск Фамилия И О (без точек)
    def has_valid_full_fio(fio_cell):
        if not fio_cell:
            return False
        for fio in fio_cell.split('\n'):
            parts = fio.strip().split()
            if len(parts) == 3 and all(re.fullmatch(r'[А-ЯЁ][а-яё]+', p) for p in parts):
                return True
        return False

    for idx, row in temp_df.iterrows():
        if not has_valid_full_fio(row["ФИО или Номер"]):
            raw_text = row["Обращение"]
            extra_match = re.search(r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ])\s+([А-ЯЁ])\b', raw_text)
            if extra_match:
                candidate = ' '.join(extra_match.groups())
                if is_valid_fio_phrase(candidate):
                    current = row["ФИО или Номер"]
                    if current:
                        existing = set(current.split('\n'))
                        if candidate not in existing:
                            temp_df.at[idx, "ФИО или Номер"] = current + '\n' + candidate
                    else:
                        temp_df.at[idx, "ФИО или Номер"] = candidate

    # 2) Разносим несколько ФИО по отдельным строкам
    rows_expanded = []
    for _, row in temp_df.iterrows():
        fio_cell = row["ФИО или Номер"]
        if fio_cell:
            fio_list = [f.strip() for f in fio_cell.split('\n') if f.strip()]
            for fio in fio_list:
                rows_expanded.append({
                    "Обращение": row["Обращение"],
                    "ФИО или Номер": fio
                })
        else:
            rows_expanded.append({
                "Обращение": row["Обращение"],
                "ФИО или Номер": ""
            })

    # 3) Удаляем дубли
    result_df = pd.DataFrame(rows_expanded).drop_duplicates(
        subset=["Обращение", "ФИО или Номер"], keep="first"
    ).reset_index(drop=True)

    return result_df

# === API ENDPOINT ===
@app.route('/process', methods=['POST'])
def process_api():
    try:
        # Получаем данные из POST-запроса
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Создаём DataFrame из входных данных
        df_input = pd.DataFrame(data)

        # Переименовываем первый столбец в "text" (если он не называется так)
        if "text" not in df_input.columns:
            df_input.rename(columns={df_input.columns[0]: "text"}, inplace=True)

        # Обрабатываем данные
        result_df = process_data(df_input)

        # Возвращаем результат в формате JSON
        return jsonify(result_df.to_dict(orient='records')), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)