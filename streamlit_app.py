import streamlit as st
import os, io, tempfile, fitz, pikepdf, multiprocessing
from pikepdf.models.image import PdfImage
from pikepdf import Name
from PIL import Image, ImageEnhance

Image.LOAD_TRUNCATED_IMAGES = True

def get_raw_stream_length(obj):
    try:
        if '/Length' in obj: return int(obj['/Length'])
    except: pass
    try: return len(obj.read_raw_bytes())
    except: pass
    return 0

def pil_from_pdfimage(obj):
    try:
        pdf_img = PdfImage(obj)
        return pdf_img.as_pil_image()
    except: return None

def resize_image(pil_img, max_dim=None):
    if not max_dim: return pil_img
    w, h = pil_img.size
    if w <= max_dim and h <= max_dim: return pil_img
    if w > h:
        nw = max_dim
        nh = int(h * (max_dim / w))
    else:
        nh = max_dim
        nw = int(w * (max_dim / h))
    return pil_img.resize((nw, nh), Image.Resampling.LANCZOS)

def remove_text_watermark_fitz(input_path, output_path, text_to_remove):
    try:
        doc = fitz.open(input_path)
        found_any = False
        phrases = [p.strip() for p in text_to_remove.split(',')]
        for page in doc:
            page_hits = 0
            for phrase in phrases:
                if not phrase: continue
                quads = page.search_for(phrase)
                if quads:
                    page_hits += 1
                    found_any = True
                    for quad in quads:
                        page.add_redact_annot(quad, fill=False)
            if page_hits > 0:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_NONE)
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        return True, found_any
    except Exception as e: return False, str(e)

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False, progress_queue=None):
    try:
        src_doc = fitz.open(input_pdf_path)
        out_doc = fitz.open()
        if fax_mode: zoom = 2.5
        else:
            zoom = 1.0 + (quality / 100.0)
            if zoom > 2.0: zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        total_pages = len(src_doc)
        for i, page in enumerate(src_doc):
            if progress_queue: progress_queue.put(('progress', i + 1, total_pages))
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB"
            if pix.n == 1: mode = "L"
            elif pix.n == 4: mode = "CMYK"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            if fax_mode:
                if img.mode != 'L': img = img.convert('L')
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)
                img = img.point(lambda x: 0 if x < 128 else 255, '1')
                img.save(buf, format="TIFF", compression="group4")
            else:
                if grayscale: img = img.convert('L')
                elif mode == 'RGB':
                    try:
                        sample = img.resize((64, 64), resample=Image.NEAREST)
                        if sample.mode == 'RGB':
                            extrema = sample.getextrema()
                            if len(extrema) == 3 and extrema[0] == extrema[1] == extrema[2]:
                                img = img.convert('L')
                    except: pass
                img.save(buf, format="JPEG", quality=int(quality), optimize=True, progressive=True, subsampling=2)
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=buf.getvalue())
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close()
        out_doc.close()
        return True, "Rasterization successful"
    except Exception as e: return False, f"Rasterization failed: {e}"

def yield_images_from_resources(resources, processed_oids):
    if '/XObject' not in resources: return
    xobjs = resources['/XObject']
    for name, xobj_ref in xobjs.items():
        try:
            xobj = xobj_ref
            oid = getattr(xobj, 'objid', None)
            if oid is not None and oid in processed_oids: continue
            subtype = xobj.get('/Subtype')
            if subtype == pikepdf.Name('/Image'):
                if oid is not None: processed_oids.add(oid)
                yield xobj
            elif subtype == pikepdf.Name('/Form'):
                if oid is not None: processed_oids.add(oid)
                if '/Resources' in xobj:
                    yield from yield_images_from_resources(xobj['/Resources'], processed_oids)
        except: continue

def process_pdf_pipeline(args):
    input_path, output_path, quality_val, mode, grayscale, watermark_text, progress_queue = args
    temp_cleaned_path = None
    try:
        current_input = input_path
        if watermark_text and watermark_text.strip():
            fd, temp_cleaned_path = tempfile.mkstemp(suffix="_wm.pdf")
            os.close(fd)
            if progress_queue: progress_queue.put(('status', f"DEBUG: Watermark Removal '{watermark_text}'...", 0))
            success, msg = remove_text_watermark_fitz(input_path, temp_cleaned_path, watermark_text)
            if success:
                current_input = temp_cleaned_path
                if progress_queue: progress_queue.put(('status', "DEBUG: Watermark removal done.", 0))

        if mode.startswith('rasterize'):
            if progress_queue: progress_queue.put(('status', f"DEBUG: Rasterizing ({mode})...", 0))
            return rasterize_and_rebuild(current_input, output_path, quality_val, grayscale, fax_mode=(mode == 'rasterize_fax'), progress_queue=progress_queue)

        if progress_queue: progress_queue.put(('status', "DEBUG: Structural Cleaning...", 0))
        pdf = pikepdf.open(current_input, allow_overwriting_input=True)
        try: pdf.docinfo.clear()
        except: pass
        try:
            if '/Outlines' in pdf.Root: del pdf.Root['/Outlines']
        except: pass
        for page in pdf.pages:
            if '/Annots' in page: del page['/Annots']
            if '/AA' in page: del page['/AA']
        if '/AA' in pdf.Root: del pdf.Root['/AA']
        if '/OpenAction' in pdf.Root: del pdf.Root['/OpenAction']

        max_dim = 1500 if mode == 'aggressive' else None
        total_pages = len(pdf.pages)
        processed_oids = set()
        for page_idx, page in enumerate(pdf.pages):
            if progress_queue: progress_queue.put(('progress', page_idx + 1, total_pages))
            if '/Resources' not in page: continue
            for obj in yield_images_from_resources(page['/Resources'], processed_oids):
                try:
                    has_mask = False
                    for k in ['/Mask', '/SMask', '/ImageMask', '/Matte']:
                        if k in obj:
                            has_mask = True
                            break
                    if has_mask: continue
                    if '/Decode' in obj: continue
                    cs = obj.get('/ColorSpace')
                    if not (cs == pikepdf.Name('/DeviceRGB') or cs == pikepdf.Name('/DeviceGray')): continue
                    current_filter = obj.get('/Filter')
                    if current_filter:
                        filters = current_filter if isinstance(current_filter, list) else [current_filter]
                        if any(f in (pikepdf.Name('/CCITTFaxDecode'), pikepdf.Name('/JBIG2Decode'), pikepdf.Name('/JPXDecode')) for f in filters): continue
                    if obj.get('/BitsPerComponent') == 1: continue

                    img = pil_from_pdfimage(obj)
                    if img is None or img.mode in ('1', 'CMYK', 'P') or img.mode not in ('RGB', 'L'): continue
                    original_compressed_size = get_raw_stream_length(obj)

                    if grayscale: new_img = img.convert("L") if img.mode != "L" else img
                    else: new_img = img if img.mode == "L" else img.convert("RGB")
                    new_mode = "DeviceGray" if new_img.mode == "L" else "DeviceRGB"
                    if max_dim: new_img = resize_image(new_img, max_dim)

                    new_data = None
                    if mode == 'lossless-smart':
                        buf = io.BytesIO()
                        new_img.save(buf, format="PNG", optimize=True)
                        if original_compressed_size > 0 and len(buf.getvalue()) < original_compressed_size:
                            new_data = buf.getvalue()
                            new_filter = pikepdf.Name("/FlateDecode")
                    else:
                        buf = io.BytesIO()
                        new_img.save(buf, format="JPEG", quality=quality_val, subsampling=(0 if mode == 'safe' else 2), optimize=True)
                        if original_compressed_size > 0 and len(buf.getvalue()) < original_compressed_size:
                            new_data = buf.getvalue()
                            new_filter = pikepdf.Name("/DCTDecode")

                    if new_data:
                        obj.write(new_data, filter=new_filter)
                        obj["/Type"] = pikepdf.Name("/XObject")
                        obj["/Subtype"] = pikepdf.Name("/Image")
                        obj["/Width"] = new_img.width
                        obj["/Height"] = new_img.height
                        obj["/ColorSpace"] = pikepdf.Name("/" + new_mode)
                        obj["/BitsPerComponent"] = 8
                        obj["/Length"] = len(new_data)
                        for k in list(obj.keys()):
                            if k not in {'/Type', '/Subtype', '/Width', '/Height', '/ColorSpace', '/BitsPerComponent', '/Length', '/Filter'}: del obj[k]
                except: continue
        pdf.save(output_path, object_stream_mode=pikepdf.ObjectStreamMode.generate)
        pdf.close()
        return True, "Success"
    except Exception as e: return False, f"Error: {e}"
    finally:
        if temp_cleaned_path and os.path.exists(temp_cleaned_path):
            try: os.remove(temp_cleaned_path)
            except: pass

class StreamlitLogger:
    def __init__(self, log_placeholder, progress_bar):
        self.log_placeholder = log_placeholder
        self.progress_bar = progress_bar
    def put(self, msg):
        if msg[0] == 'progress':
            _, current, total = msg
            if total > 0: self.progress_bar.progress(current / total)
        elif msg[0] == 'status': self.log_placeholder.code(msg[1])

def main():
    st.set_page_config(page_title="CleanPDF", layout="centered")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    with st.expander("⚙️ Configuration", expanded=True):
        enable_wm = st.checkbox("Remove Text")
        wm_text = st.text_input("Text to remove") if enable_wm else ""
        enable_compress = st.checkbox("Enable Compression")
        mode, quality, grayscale = "safe", 100, False
        if enable_compress:
            mode_disp = st.selectbox("Mode", ["Safe Compression", "Aggressive Compression", "Lossless Smart", "Rasterize (Standard)", "Rasterize (B&W Fax Mode)"])
            mode = {"Safe Compression": "safe", "Aggressive Compression": "aggressive", "Lossless Smart": "lossless-smart", "Rasterize (Standard)": "rasterize", "Rasterize (B&W Fax Mode)": "rasterize_fax"}[mode_disp]
            if "Rasterize" in mode_disp or mode_disp in ["Safe Compression", "Aggressive Compression"]: quality = st.slider("Quality", 10, 100, 75)
            grayscale = st.checkbox("Grayscale")

    if st.button("Start", type="primary", disabled=not uploaded_files):
        pbar = st.progress(0)
        log = st.empty()
        logger = StreamlitLogger(log, pbar)
        results = []
        for up_file in uploaded_files:
            logger.put(('status', f"Processing: {up_file.name}...", 0))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as ti, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as to:
                ti.write(up_file.getvalue())
                input_p, output_p = ti.name, to.name
            success, msg = process_pdf_pipeline((input_p, output_p, quality, mode, grayscale, wm_text, logger))
            if success:
                with open(output_p, "rb") as f: results.append({"name": f"clean_{up_file.name}", "data": f.read()})
                logger.put(('status', f"Done: {up_file.name}", 0))
            else: st.error(f"Error {up_file.name}: {msg}")
            try: os.remove(input_p); os.remove(output_p)
            except: pass
        pbar.progress(100)
        st.success("Done!")
        for res in results: st.download_button(f"Download {res['name']}", res['data'], res['name'], "application/pdf")

if __name__ == "__main__": main()
