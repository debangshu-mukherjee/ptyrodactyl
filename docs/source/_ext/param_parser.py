from sphinx.ext.napoleon.docstring import NumpyDocstring


class CustomNumpyDocstring(NumpyDocstring):
    def _parse_parameters_section(self, section):
        items = []
        for line in section.splitlines():
            if line.strip().startswith("-"):
                param_line = line.strip("- ").split(":", 1)
                if len(param_line) == 2:
                    param_str = param_line[0].strip()
                    name_type = param_str.split("(", 1)
                    if len(name_type) == 2:
                        name = name_type[0].strip("`")
                        type_ = "(" + name_type[1].rstrip(")")
                        desc = param_line[1].strip()
                        items.append((name, type_, desc))
        return items


def process_docstring(app, what, name, obj, options, lines) -> None:
    docstring = "\n".join(lines)
    custom_doc = CustomNumpyDocstring(docstring)
    lines[:] = str(custom_doc).splitlines()


# This needs to be at the module level
def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
    return {"version": "1.0", "parallel_read_safe": True}
