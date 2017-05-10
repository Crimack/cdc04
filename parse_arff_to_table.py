# coding: utf-8
import os

data = []
TARGET_PATH = os.path.join(os.getcwd(), "experiments", "test_files")
for filename in os.listdir(TARGET_PATH):
    if filename.endswith(".arff"):
	    with open(os.path.join(TARGET_PATH, filename)) as f:
		info = {"name": filename.replace("_", "-")[:-5], "num_atts": 0, "num_instances": 0}
		for line in f.readlines():
		    if line.startswith("@attribute"):
			info["num_atts"] += 1
		    elif (not line.startswith("@relation")) and (not line.startswith("@data")) and line != "\n":
			info["num_instances"] += 1
		data.append(info)
        

OUTPUT_PATH = os.path.join(os.getcwd(), "doc", "dissertation", "datasets_table.tex")
with open(OUTPUT_PATH, "w") as g:
    g.write("{\\centering \\footnotesize \\begin{longtable}{lrr@{\hspace{0.1cm}}cr@{\hspace{0.1cm}}c}\n")
    g.write("\caption{\label{dsets} Experimental Data Sets}\n")
    g.write("\\\\\n")
    g.write("\hline\\\\\n")
    g.write("Dataset Title & Num. Attributes & Num. Instances\\\\\n")
    g.write("\hline\\\\\n")
    for entry in sorted(data, key=lambda k: k["name"]):
        g.write("{name} & {num_atts} & {num_instances}\\\\\n".format(**entry))
    g.write("\hline\\\\\n")
    g.write("\end{longtable} \\footnotesize \par}\n")
    g.write("\\newpage\n")
