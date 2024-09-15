r"""
jupyter nbconvert --to Markdown sundial.ipynb produces markdown with latex output like this:

$\displaystyle
        \begin{equation}
            f_1 \wedge f_2 = \cos{\left (\alpha  \right )} \boldsymbol{e}_{1}\wedge \boldsymbol{e}_{2} - \sin{\left (\alpha  \right )} \boldsymbol{e}_{2}\wedge \boldsymbol{e}_{3} \nonumber
        \end{equation}
        $

However, environments such as equation and align require math mode, which this script
hacks into place within the ipynb file before converting to markdown.
"""

import json
import sys

if len(sys.argv) < 3:
    raise Exception("Usage: python dollar_dollar.py in.ipynb out.ipynb")

with open(sys.argv[1], mode="r", encoding="utf-8") as f:
    doc = json.loads(f.read())

for cell in doc["cells"]:
    if cell["cell_type"] == "code":
        for output in cell["outputs"]:
            if "data" in output.keys():
                if "text/latex" in output["data"]:
                    latex_lines = output["data"]["text/latex"]
                    if latex_lines[0].startswith("$") and latex_lines[-1].endswith("$"):
                        latex_lines[0] = "$" + latex_lines[0]
                        latex_lines[-1] = latex_lines[-1] + "$"

with open(sys.argv[2], mode="w") as f:
    f.write(json.dumps(doc))
