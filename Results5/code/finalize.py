from pathlib import Path
import fitz,re
import os
root=Path(os.environ["FIG5_WORK"])
output=Path(__file__).resolve().parents[1]/"figures"
src=fitz.open(root/'Fig5_bars.pdf')
p=src[0];scale=180*72/25.4/p.rect.width
spans=[(line['dir'],s) for b in p.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans'] if s['text'].strip()]
legend_spans = [s for _, s in spans if "(within-disease, within-library GSEA FDR)" in s["text"]]
assert len(legend_spans) == 1
legend = legend_spans[0]
legend["text"] = legend["text"].split("(within-disease",1)[0].rstrip()
streams=set(p.get_contents())|{x[0] for x in p.get_xobjects()}
for x in streams:
    buf=src.xref_stream(x)
    if buf and b'BT' in buf:
        src.update_stream(x,re.sub(rb'(?<!\S)BT\b.*?\bET(?!\S)',b'',buf,flags=re.S))
assert not p.get_text().strip()
out=fitz.open();page=out.new_page(width=p.rect.width*scale,height=p.rect.height*scale)
page.show_pdf_page(page.rect,src,0)
fonts={}
for name,fn in [('Reg','arial'),('Bold','arialbd'),('Italic','ariali')]:
    path=f'C:/Windows/Fonts/{fn}.ttf'
    fonts[name]=fitz.Font(fontfile=path)
    page.insert_font(fontname=name,fontfile=path)
for direction,s in spans:
    txt=s['text'];face='Bold' if 'Bold' in s['font'] else 'Italic' if 'Oblique' in s['font'] else 'Reg'
    font=fonts[face];size=max(5.05,min(8,s['size']*scale))
    if len(txt)==1 and txt in 'abcdefghij' and face=='Bold':size=7.5
    submap=str.maketrans('₀₁₂','012')
    assert all(font.has_glyph(ord(c)) for c in txt.translate(submap)),repr(txt)
    x,y=s['origin'];box=fitz.Rect(s['bbox']);w=font.text_length(txt.translate(submap),fontsize=size)
    rotate=0
    if direction[1]<-.9:
        rotate=90
        pos=(x*scale,box.y0*scale+(box.height*scale+w)/2)
    else:
        center=(box.x0+box.x1)*scale/2
        left=center-w/2
        if 636<box.y0<829 and any(abs(box.x1-edge)<2 for edge in [188,548,908]):left=box.x1*scale-w
        pos=(max(0.75,left),y*scale)
    if box.y0>900 and txt in ('Nonsense-mediated','mRNA decay','GCN2 amino acid','deficiency response'):
        pieces={'Nonsense-mediated':['Nonsense-','mediated'],'mRNA decay':['mRNA decay'],'GCN2 amino acid':['GCN2 amino','acid'],'deficiency response':['deficiency','response']}[txt]
        if txt=='deficiency response':
            pieces=['deficiency response']
        if txt in ('mRNA decay','deficiency response'):
            pos=(pos[0],pos[1]+5.4)
        for i,piece in enumerate(pieces):
            page.insert_text((center-font.text_length(piece,fontsize=size)/2,pos[1]+5.4*i),piece,fontname=face,fontsize=size,color=fitz.sRGB_to_pdf(s['color']))
        continue
    for part in re.findall(r'[₀₁₂]+|[^₀₁₂]+',txt):
        shift=1.15 if part[0] in '₀₁₂' else 0
        pt=(pos[0]+(shift if rotate else 0),pos[1]+(0 if rotate else shift))
        part=part.translate(submap)
        page.insert_text(pt,part,fontname=face,fontsize=size,color=fitz.sRGB_to_pdf(s['color']),rotate=rotate)
        adv=font.text_length(part,fontsize=size)
        pos=(pos[0]+(0 if rotate else adv),pos[1]-(adv if rotate else 0))
out.subset_fonts()
dest=output/'Figure5.pdf';out.save(dest,garbage=4,deflate=True)
with fitz.open(dest) as d:
    page=d[0]
    visible=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
    print(len(spans),len(visible),sorted(set(s['font'] for s in visible)),min(s['size'] for s in visible),max(s['size'] for s in visible))
    assert all(5<=s['size']<=8.01 and 'Arial' in s['font'] for s in visible)
    def norm(t):return ''.join(t.split()).translate(submap).replace(chr(173),'-')
    from collections import Counter
    assert Counter(norm(''.join(s['text'] for _,s in spans)))==Counter(norm(''.join(s['text'] for s in visible)))
    page.get_pixmap(dpi=600).save(output/'Figure5.png')






