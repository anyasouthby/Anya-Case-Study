from ingestion import load_dataset


documents = load_dataset("data/dataset")

for document in documents:
    print("\n" + "=" * 80)
    print(document["filename"])
    print("=" * 80)
    print(document["text"][:500])