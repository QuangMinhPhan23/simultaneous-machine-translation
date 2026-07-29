"""
Build a sentence-level test set from one dialect/domain slice of the UBC-NLP/alexandria
dataset, using the same {"source", "reference", "src_lang", "tgt_lang"} schema as the WMT
test files. Each conversation turn becomes one test example.
"""
import argparse
import json
import os

from datasets import load_dataset

DIALECT_NAMES = {
    "EG": "Egyptian Arabic",
    "SA": "Saudi Arabic",
    "LB": "Lebanese Arabic",
    "MA": "Moroccan Arabic",
    "JO": "Jordanian Arabic",
    "PS": "Palestinian Arabic",
    "SY": "Syrian Arabic",
    "OM": "Omani Arabic",
    "YE": "Yemeni Arabic",
    "SD": "Sudanese Arabic",
    "LY": "Libyan Arabic",
    "MR": "Mauritanian Arabic",
    "TN": "Tunisian Arabic",
}


def build_test_set(country, domain, split, direction, tgt_label=None):
    dataset = load_dataset("UBC-NLP/alexandria", name=country, split=split)
    if domain is not None:
        dataset = dataset.filter(lambda x: x["domain"] == domain)

    tgt_dialect_name = tgt_label or DIALECT_NAMES.get(country, f"Arabic ({country})")

    examples = []
    for conv in dataset:
        for eng_turn, dia_turn in zip(conv["english_conversation"], conv["dialectal_conversation"]):
            eng_text = eng_turn["text"].strip()
            dia_text = dia_turn["text"].strip()
            if not eng_text or not dia_text:
                continue

            if direction == "en2ar":
                source, reference = eng_text, dia_text
                src_lang, tgt_lang = "English", tgt_dialect_name
            else:
                source, reference = dia_text, eng_text
                src_lang, tgt_lang = tgt_dialect_name, "English"

            examples.append(
                {
                    "source": source,
                    "reference": reference,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "conv_id": conv["conv_id"],
                    "domain": conv["domain"],
                }
            )
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="EG", help="Alexandria country/dialect code")
    parser.add_argument("--domain", default="Commerce and transactions",
                         help="Conversation domain to filter to; pass empty string for all domains")
    parser.add_argument("--split", default="test")
    parser.add_argument("--direction", choices=["en2ar", "ar2en"], default="en2ar")
    parser.add_argument("--tgt_label", default=None,
                         help="Override the tgt_lang string used in the instruction prompt; "
                              "source and reference text stay the same")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    domain = args.domain or None
    examples = build_test_set(args.country, domain, args.split, args.direction, args.tgt_label)

    if args.output is None:
        domain_tag = domain.lower().replace(" ", "_") if domain else "all_domains"
        label_tag = "" if not args.tgt_label else "." + args.tgt_label.lower().split(" ")[0].strip("(),")
        args.output = (
            f"data/mt_data/test_data/alexandria.{args.split}.{args.country.lower()}."
            f"{domain_tag}.{args.direction}{label_tag}.json"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(examples)} examples to {args.output}")
