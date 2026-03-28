"""
Helper utilities to create html reports.
"""

import html
from contextlib import contextmanager


class Base:
    """Base HTML output"""

    def __init__(self, f):
        self._f = f

    def print(self, text):
        print(text, file=self._f)

    @contextmanager
    def tag(self, name):
        self.print(f"<{name}>")
        yield self
        self.print(f"</{name}>")


class Document(Base):
    def __init__(self, f: str):
        super().__init__(f)

    def __enter__(self):
        self.print("<!DOCTYPE html><html>")
        return self

    def __exit__(self, *exc):
        self.print("</html>")

    def head(self) -> "Head":
        return Head(self._f)

    def body(self) -> "Body":
        return Body(self._f)


class Head(Base):
    def __enter__(self):
        self.print("<head>")
        return self

    def __exit__(self, *exc):
        self.print("</head>")

    def title(self, caption: str):
        with self.tag("title"):
            self.print(caption)

    def style(self, content: str):
        with self.tag("style"):
            self.print(content)


class Body(Base):
    """HTML body"""

    def __enter__(self):
        self.print("<body>")
        return self

    def __exit__(self, *exc):
        self.print("</body>")

    def h1(self, caption: str):
        self.header(caption, level=1)

    def h2(self, caption: str):
        self.header(caption, level=2)

    def h3(self, caption: str):
        self.header(caption, level=3)

    def h4(self, caption: str):
        self.header(caption, level=4)

    def header(self, caption: str, level=1):
        with self.tag(f"h{level}"):
            self.print(caption)

    def paragraph(self, text: str):
        with self.tag("p"):
            self.print(text)

    def pre(self, text: str):
        with self.tag("pre"):
            self.print(html.escape(text))

    def table(self) -> "Table":
        return Table(self._f)


class Table(Base):
    """Table item"""

    def __enter__(self):
        self.print('<table border="1">')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.print("</table>")

    def header(self, *texts):
        self.row(*texts, tag="th")

    def row(self, *texts, tag="td", escape=True):
        with self.tag("tr"):
            for text in texts:
                with self.tag(tag):
                    text = html.escape(text) if escape else text
                    self.print(text)
