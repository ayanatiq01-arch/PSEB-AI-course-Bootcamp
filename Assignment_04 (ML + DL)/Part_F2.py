from transformers import pipeline
import pandas as pd

# Load pretrained sentiment model
sentiment = pipeline("sentiment-analysis")

# Your sentences and manually assigned labels
reviews = [
    ("English", "The biryani at Burns Road was delicious and full of flavour.", "POSITIVE"),
    ("English", "The service at this restaurant was extremely slow.", "NEGATIVE"),
    ("English", "The food was not bad at all, actually quite tasty.", "POSITIVE"),
    ("English", "The chai at this cafe was amazing.", "POSITIVE"),
    ("English", "The restaurant looked beautiful, but the food was disappointing.", "NEGATIVE"),
    ("English", "I absolutely loved the burgers at this place.", "POSITIVE"),
    ("English", "Great, another restaurant that serves cold food!", "NEGATIVE"),
    ("English", "The sea view at Clifton was lovely, but the snacks were average.", "POSITIVE"),
    ("English", "The biryani was too oily and had very little flavour.", "NEGATIVE"),
    ("English", "The food was excellent and the staff were friendly.", "POSITIVE"),

    ("Roman Urdu", "Burns Road ki biryani bohat zabardast thi.", "POSITIVE"),
    ("Roman Urdu", "Restaurant ka khana bilkul acha nahi tha.", "NEGATIVE"),
    ("Roman Urdu", "Chai itni achi thi ke maza aa gaya.", "POSITIVE"),
    ("Roman Urdu", "Wah, kya service hai, aik ghanta wait karwaya!", "NEGATIVE"),
    ("Roman Urdu", "Food theek tha lekin prices bohat zyada thay.", "NEGATIVE"),
    ("Roman Urdu", "Clifton ka view acha tha aur snacks bhi tasty thay.", "POSITIVE"),
    ("Roman Urdu", "Biryani buri nahi thi, dobara kha sakta hoon.", "POSITIVE"),
    ("Roman Urdu", "Burger acha tha magar fries bilkul thandi thin.", "NEGATIVE"),
    ("Roman Urdu", "Staff friendly tha aur environment bhi acha tha.", "POSITIVE"),
    ("Roman Urdu", "Itna mehnga khana aur taste zero, paisay zaya ho gaye.", "NEGATIVE")
]

results = []

for language, sentence, true_label in reviews:

    prediction = sentiment(sentence)[0]

    results.append({
        "Language": language,
        "Sentence": sentence,
        "Your Label": true_label,
        "Model Label": prediction["label"],
        "Model Score": round(prediction["score"], 4)
    })

# Display result table
df = pd.DataFrame(results)

print("\nSENTIMENT ANALYSIS RESULTS")
print(df.to_string(index=False))

# Calculate accuracy separately for each language
for language in ["English", "Roman Urdu"]:

    language_data = df[df["Language"] == language]

    correct = (
        language_data["Your Label"] ==
        language_data["Model Label"]
    ).sum()

    total = len(language_data)
    accuracy = correct / total

    print(f"\n{language} Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}/{total}")

# Find incorrect predictions
wrong = df[df["Your Label"] != df["Model Label"]]

print("\nWRONG PREDICTIONS AND TOKENIZATION")

tokenizer = sentiment.tokenizer

for index, row in wrong.head(3).iterrows():

    print("\nSentence:", row["Sentence"])

    tokens = tokenizer.tokenize(row["Sentence"])

    print("Tokens:", tokens)
    print("Your label:", row["Your Label"])
    print("Model label:", row["Model Label"])