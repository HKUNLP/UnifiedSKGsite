FILE_PATH = "../docs/benchmarks/benchmarks_old.md"
from collections import OrderedDict
with open(FILE_PATH) as f:
    lines = [line.strip() for line in f.readlines()]

table_dict = {"dataset": [], "Knowledge": [], "User Input": [], "Output": [], "Keywords": [], }
table_dict = OrderedDict(table_dict)
for line in lines:
    # print line
    if line.startswith("###"):
        dataset = line.strip("# ")
        table_dict["dataset"].append(dataset)
    if line.startswith("> - **Knowledge**:"):
        table_dict["Knowledge"].append(line.replace("> - **Knowledge**:", ""))
    if line.startswith("> - **User Input**:"):
        table_dict["User Input"].append(line.replace("> - **User Input**:", ""))
    if line.startswith("> - **Output**:"):
        table_dict["Output"].append(line.replace("> - **Output**:", ""))
    if line.startswith("> - **Keywords**:"):
        table_dict["Keywords"].append(line.replace("> - **Keywords**:", ""))

string = ""
for k, o, u, kn, d in zip(table_dict["Keywords"], table_dict["Output"], table_dict["User Input"], table_dict["Knowledge"], table_dict["dataset"]):
    string += "|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t\t|\n".format(d, kn, u, o, k)

print string