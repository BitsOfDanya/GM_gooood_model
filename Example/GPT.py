from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

def parse_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    themes = []
    categories = []

    current_section = None
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "Тема":
            current_section = "theme"
            continue
        elif stripped_line == "Категория":
            current_section = "category"
            continue

        if current_section == "theme":
            themes.append(stripped_line)
        elif current_section == "category":
            categories.append(stripped_line)

    theme_category_mapping = dict(zip(themes, categories))
    return theme_category_mapping


def predict_category_and_theme(description, theme_category_mapping):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    themes_list = list(theme_category_mapping.keys())
    themes_text = ", ".join(themes_list)
    prompt_text = f"Given the website description: '{description}', which among the following themes is most relevant? Themes: {themes_text}. Please respond with only the relevant theme."

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, top_k=50, top_p=0.95, temperature=0.7)

    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    predicted_theme = predicted_text.split(prompt_text)[1].strip()

    if predicted_theme not in themes_list:
        return "Unknown", "Unknown"

    predicted_category = theme_category_mapping[predicted_theme]
    return predicted_category, predicted_theme

theme_category_mapping = parse_file("text.txt")
description = "www.babykrd.ru. expecting-parents. healthy-lifestyle. Здоровый образ жизни - Официальный сайт ГК) ДРС Nº 1 M3 KK. Sep 6, 2019. Российская Федерация, Краснодарский край, город Краснодар, улица Ставропольская, 155 тел./факс.: (861)233-63-35, email: krsbabyh@miackuban.ru."
predicted_category, predicted_theme = predict_category_and_theme(description, theme_category_mapping)

print(f"Category: {predicted_category}")
print(f"Theme: {predicted_theme}")
