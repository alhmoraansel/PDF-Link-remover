import streamlit as st
import os
import io
import tempfile
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from PIL import Image

# --- CORE LOGIC (UNCHANGED) ---
Image.LOAD_TRUNCATED_IMAGES = True

def is_filter(obj, name_str):
    try:
        f = obj.get('/Filter')
        return f == pikepdf.Name(name_str)
    except Exception: return False

def read_stream_bytes(obj):
    try: return obj.get_stream_buffer()
    except Exception: pass
    try: return obj.read_bytes()
    except Exception: pass
    return b''

def pil_from_pdfimage(pdf_img):
    pil_img = pdf_img.as_pil_image()
    if pil_img.mode == 'CMYK': pil_img = pil_img.convert('RGB')
    if pil_img.mode == '1': pil_img = pil_img.convert('L').convert('RGB')
    return pil_img

def flatten_alpha(pil_img, background_color=(255,255,255)):
    if pil_img.mode in ('RGBA', 'LA'):
        alpha = pil_img.split()[-1]
        base = Image.new('RGB', pil_img.size, background_color)
        base.paste(pil_img, mask=alpha)
        return base
    elif pil_img.mode == 'P' and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')
        alpha = pil_img.split()[-1]
        base = Image.new('RGB', pil_img.size, background_color)
        base.paste(pil_img, mask=alpha)
        return base
    return pil_img.convert('RGB')

def jpeg_bytes_from_pil(pil_img, quality, subsampling=None):
    buf = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None: save_kwargs["subsampling"] = int(subsampling)
    pil_img.save(buf, **save_kwargs)
    return buf.getvalue()

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False):
    try:
        src_doc = fitz.open(input_pdf_path)
        out_doc = fitz.open()
        zoom = 2.5 if fax_mode else (1.0 + (quality / 100.0))
        if zoom > 2.0: zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        for page in src_doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB" if pix.n >= 3 else "L", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            if fax_mode:
                img = img.convert('L').point(lambda x: 0 if x < 200 else 255, '1')
                img.save(buf, format="TIFF", compression="group4")
            else:
                if grayscale: img = img.convert('L')
                img.save(buf, format="JPEG", quality=int(quality), optimize=True)
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=buf.getvalue())
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close(); out_doc.close()
        return True, "Rasterization successful"
    except Exception as e: return False, str(e)

def process_pdf_pipeline(input_path, output_path, watermark_text, quality_val, mode, grayscale):
    fd1, temp_rep = tempfile.mkstemp(suffix=".pdf"); os.close(fd1)
    fd2, temp_cln = tempfile.mkstemp(suffix=".pdf"); os.close(fd2)
    try:
        with pikepdf.open(input_path) as pdf:
            pdf.save(temp_rep, fix_metadata_version=True)
        
        doc = fitz.open(temp_rep)
        doc.set_metadata({})
        for page in doc:
            if watermark_text:
                for rect in page.search_for(watermark_text): page.add_redact_annot(rect)
                page.apply_redactions(images=0, graphics=0)
            page.clean_contents()
        doc.save(temp_cln, garbage=4, deflate=True)
        doc.close()

        if 'rasterize' in mode:
            return rasterize_and_rebuild(temp_cln, output_path, quality_val, grayscale, mode=='rasterize_fax')

        with pikepdf.open(temp_cln) as pdf:
            pdf.docinfo.clear()
            for obj in pdf.objects:
                if isinstance(obj, pikepdf.Stream) and obj.get('/Subtype') == '/Image':
                    try:
                        pdf_img = PdfImage(obj)
                        pil_img = pil_from_pdfimage(pdf_img)
                        if grayscale: pil_img = pil_img.convert('L')
                        
                        if mode == 'aggressive':
                            img_bytes = jpeg_bytes_from_pil(flatten_alpha(pil_img), quality_val)
                        else:
                            img_bytes = jpeg_bytes_from_pil(pil_img, quality_val)
                            
                        obj.write(img_bytes)
                        obj['/Filter'] = pikepdf.Name('/DCTDecode')
                    except: continue
            pdf.save(output_path)
        return True, "Compression Complete"
    except Exception as e: return False, str(e)
    finally:
        for p in [temp_rep, temp_cln]:
            if os.path.exists(p): os.remove(p)

# --- STREAMLIT APP ---
st.set_page_config(page_title="PDF Crusher", page_icon="🛡️")
st.title("🛡️ PDF Processor")

if "out_bytes" not in st.session_state: st.session_state.out_bytes = None
if "out_name" not in st.session_state: st.session_state.out_name = ""

with st.sidebar:
    st.header("Settings")
    mode_choice = st.selectbox("Mode", ["lossless-smart", "safe", "aggressive", "rasterize", "rasterize_fax"])
    quality = st.slider("Quality", 10, 100, 70)
    grayscale = st.checkbox("Grayscale")
    watermark = st.text_input("Redact Text", "")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.button("PROCESS FILE"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                tmp_in.write(uploaded_file.getbuffer())
                in_path = tmp_in.name
            
            out_path = in_path + "_out.pdf"
            success, msg = process_pdf_pipeline(in_path, out_path, watermark, quality, mode_choice, grayscale)
            
            if success:
                with open(out_path, "rb") as f:
                    st.session_state.out_bytes = f.read()
                st.session_state.out_name = f"processed_{uploaded_file.name}"
                st.success(msg)
            else:
                st.error(msg)
            
            if os.path.exists(in_path): os.remove(in_path)
            if os.path.exists(out_path): os.remove(out_path)

if st.session_state.out_bytes:
    st.download_button(
        label="📥 DOWNLOAD NOW",
        data=st.session_state.out_bytes,
        file_name=st.session_state.out_name,
        mime="application/pdf"
    )
