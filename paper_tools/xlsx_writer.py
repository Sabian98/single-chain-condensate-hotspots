"""
xlsx_writer.py: minimal .xlsx writer with the standard library only (no openpyxl on this Mac). Writes one or more sheets of
rows, numbers as numbers, text as inline strings, a bold frozen header row, autofilter and simple column widths.
  write_xlsx(path, [(sheet_name, header, rows), ...])
"""
import zipfile
from xml.sax.saxutils import escape


def _col(i):
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref, v, style):
    st = f' s="{style}"' if style else ''
    if v is None or v == '' or (isinstance(v, float) and v != v):
        return f'<c r="{ref}"{st}/>'
    if isinstance(v, bool):
        v = 'yes' if v else 'no'
    if isinstance(v, (int, float)):
        return f'<c r="{ref}"{st}><v>{v:.6g}</v></c>' if isinstance(v, float) else f'<c r="{ref}"{st}><v>{v}</v></c>'
    return f'<c r="{ref}"{st} t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'


def _sheet(header, rows, widths):
    cols = ''.join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
    out = ['<sheetData>']
    for r, row in enumerate([header] + rows):
        cells = ''.join(_cell(f'{_col(c)}{r + 1}', v, 1 if r == 0 else 0) for c, v in enumerate(row))
        out.append(f'<row r="{r + 1}">{cells}</row>')
    out.append('</sheetData>')
    last = f'{_col(len(header) - 1)}{len(rows) + 1}'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
            f'<cols>{cols}</cols>' + ''.join(out) + f'<autoFilter ref="A1:{last}"/></worksheet>')


def write_xlsx(path, sheets):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml',
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   + ''.join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                             'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                             for i in range(len(sheets))) + '</Types>')
        z.writestr('_rels/.rels',
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                   'Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + ''.join(f'<sheet name="{escape(n)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, (n, _, _) in enumerate(sheets))
                   + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + ''.join(f'<Relationship Id="rId{i + 1}" '
                             'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                             f'Target="worksheets/sheet{i + 1}.xml"/>' for i in range(len(sheets)))
                   + f'<Relationship Id="rId{len(sheets) + 1}" '
                   'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   '</Relationships>')
        z.writestr('xl/styles.xml',
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<fonts count="2"><font><sz val="11"/><name val="Arial"/></font><font><b/><sz val="11"/><name val="Arial"/></font></fonts>'
                   '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
                   '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
                   '<cellXfs count="2"><xf fontId="0"/><xf fontId="1" applyFont="1"/></cellXfs></styleSheet>')
        for i, (_, header, rows) in enumerate(sheets):
            widths = [min(60, max(8, max(len(str(r[c])) if c < len(r) and r[c] is not None else 0 for r in [header] + rows) * 0.95 + 2))
                      for c in range(len(header))]
            z.writestr(f'xl/worksheets/sheet{i + 1}.xml', _sheet(header, rows, widths))
