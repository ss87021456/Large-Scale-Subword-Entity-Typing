import json

with open("types_with_entity_mapping.json", "r") as f:
    data = f.read().splitlines()

with open("types_with_entity_mapping_prettier.json", "w") as f:
    f.write("[\n")
    for itr in data[:-1]:
        f.write(itr + ",\n")
    f.write(data[-1] + "\n]")
    f.write("\n")

del data
a = json.load(open("types_with_entity_mapping_prettier.json", "r"))

keys = [itr["wordnet"] for itr in a]
values = [{"defintion": itr['defintion']} for itr in a]

dic = dict(zip(keys, values))

with open("desp.json", "w") as f:
    json.dump(dic, f, sort_keys=True, indent=4)
