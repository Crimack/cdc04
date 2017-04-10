# coding: utf-8
import os

data = []
target_path = os.path.join(os.getcwd(), "Test Files")
for filename in os.listdir(target_path):
    if filename.endswith(".arff"):
	    with open(os.path.join(target_path, filename)) as f:
		info = {"name": filename.replace("_", "-")[:-5], "num_atts": 0, "num_instances": 0}
		for line in f.readlines():
		    if line.startswith("@attribute"):
			info["num_atts"] += 1
		    elif (not line.startswith("@relation")) and (not line.startswith("@data")) and line != "\n":
			info["num_instances"] += 1
		data.append(info)
        

output_path = os.path.join(os.getcwd(), "Docs", "Dissertation", "McKee", "datasets_table.tex")
with open(output_path, "w") as g:
    g.write("\\begin{table}[thb]\n")
    g.write("\caption{\label{dsets} Experimental Data Sets}\n")
    g.write("\\footnotesize")
    g.write("{\\centering \\begin{tabular}{lrr@{\hspace{0.1cm}}cr@{\hspace{0.1cm}}c}\\\\\n")
    g.write("\hline\\\\\n")
    g.write("Dataset Title & Num. Attributes & Num. Instances\\\\\n")
    g.write("\hline\\\\\n")
    for entry in sorted(data, key=lambda k: k["name"]):
        g.write("{name} & {num_atts} & {num_instances}\\\\\n".format(**entry))
    g.write("\hline\\\\\n")
    g.write("\end{tabular} \\footnotesize \par}\n")
    g.write("\end{table}")
    
