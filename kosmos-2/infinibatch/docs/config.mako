<%!
    # This is a configuration file for pdoc3, the tool we use for generating html documentation from docstrings.
    # Please look at the README.md for instruction on how to generate the documentation.
    # Template configuration. Copy over in your template directory
    # (used with --template-dir) and adapt as required.
    html_lang = 'en'
    show_inherited_members = False
    extract_module_toc_into_sidebar = True
    list_class_variables_in_index = True
    sort_identifiers = False
    show_type_annotations = True
    # Show collapsed source code block next to each item.
    # Disabling this can improve rendering speed of large modules.
    show_source_code = True
    # If set, format links to objects in online source code repository
    # according to this template. Supported keywords for interpolation
    # are: commit, path, start_line, end_line.
    #git_link_template = 'https://github.com/USER/PROJECT/blob/{commit}/{path}#L{start_line}-L{end_line}'
    #git_link_template = 'https://gitlab.com/USER/PROJECT/blob/{commit}/{path}#L{start_line}-L{end_line}'
    #git_link_template = 'https://bitbucket.org/USER/PROJECT/src/{commit}/{path}#lines-{start_line}:{end_line}'
    #git_link_template = 'https://CGIT_HOSTNAME/PROJECT/tree/{path}?id={commit}#n{start-line}'
    git_link_template = None
    # A prefix to use for every HTML hyperlink in the generated documentation.
    # No prefix results in all links being relative.
    link_prefix = ''
    # Enable syntax highlighting for code/source blocks by including Highlight.js
    syntax_highlighting = True
    # Set the style keyword such as 'atom-one-light' or 'github-gist'
    #     Options: https://github.com/highlightjs/highlight.js/tree/master/src/styles
    #     Demo: https://highlightjs.org/static/demo/
    hljs_style = 'github'
    # If set, insert Google Analytics tracking code. Value is GA
    # tracking id (UA-XXXXXX-Y).
    google_analytics = ''
    # If set, render LaTeX math syntax within \(...\) (inline equations),
    # or within \[...\] or $$...$$ or `.. math::` (block equations)
    # as nicely-formatted math formulas using MathJax.
    # Note: in Python docstrings, either all backslashes need to be escaped (\\)
    # or you need to use raw r-strings.
    latex_math = False
%>