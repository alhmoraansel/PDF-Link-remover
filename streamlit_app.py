import streamlit as st
import os
import io
import tempfile
import shutil
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from PIL import Image

# --- CORE LOGIC PRESERVED ---
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
    elif pil_img.mode != 'RGB': return pil_img.convert('RGB')
    return pil_img

def jpeg_bytes_from_pil(pil_img, quality, subsampling=None):
    buf = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None: save_kwargs["subsampling"] = int(subsampling)
    pil_img.save(buf, **save_kwargs)
    return buf.getvalue()

def png_bytes_from_pil(pil_img, optimize=True):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=bool(optimize))
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
            mode = "RGB"
            if pix.n == 1: mode = "L"
            elif pix.n == 4: mode = "CMYK"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            if fax_mode:
                img = img.convert('L').point(lambda x: 0 if x < 200 else 255, '1')
                img.save(buf, format="TIFF", compression="group4")
            else:
                if grayscale: img = img.convert('L')
                img.save(buf, format="JPEG", quality=int(quality), optimize=True)
            img_bytes = buf.getvalue()
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes)
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close(); out_doc.close()
        return True, "Rasterization successful"
    except Exception as e: return False, f"Rasterization failed: {e}"

def process_pdf_pipeline(input_path, output_path, watermark_text, quality_val, mode, grayscale):
    fd1, temp_repaired = tempfile.mkstemp(suffix=".pdf")
    os.close(fd1)
    fd2, temp_cleaned = tempfile.mkstemp(suffix=".pdf")
    os.close(fd2)
    try:
        try:
            pdf = pikepdf.open(input_path)
            pdf.save(temp_repaired, fix_metadata_version=True)
            pdf.close()
        except Exception as e: return False, f"Repair failed: {e}"

        try:
            doc = fitz.open(temp_repaired)
            doc.set_metadata({})
            for page in doc:
                if watermark_text:
                    hits = page.search_for(watermark_text)
                    for rect in hits: page.add_redact_annot(rect)
                    page.apply_redactions(images=0, graphics=0)
                page.clean_contents()
            doc.save(temp_cleaned, garbage=4, deflate=True)
            doc.close()
        except Exception as e: return False, f"Cleaning failed: {e}"

        if mode in ['rasterize', 'rasterize_fax']:
            return rasterize_and_rebuild(temp_cleaned, output_path, quality_val, grayscale, mode == 'rasterize_fax')

        # Main Compression Loop
        pdf = pikepdf.open(temp_cleaned)
        pdf.docinfo.clear()
        for obj in list(pdf.objects):
            if not (isinstance(obj, pikepdf.Stream) and obj.get('/Subtype') == pikepdf.Name('/Image')):
                continue
            
            try:
                original_bytes = read_stream_bytes(obj) or b''
                original_size = len(original_bytes)
                pdf_img = PdfImage(obj)
                pil_img = pil_from_pdfimage(pdf_img)
                
                if pil_img.width < 20 or pil_img.height < 20: continue
                if grayscale and pil_img.mode != 'L': pil_img = pil_img.convert('L')

                if mode == 'aggressive':
                    pil_proc = flatten_alpha(pil_img)
                    jpeg_bytes = jpeg_bytes_from_pil(pil_proc, quality_val, subsampling=2)
                    if original_size and len(jpeg_bytes) > original_size * 1.5: continue
                    obj.write(jpeg_bytes)
                    obj['/Filter'] = pikepdf.Name('/DCTDecode')
                    obj['/ColorSpace'] = pikepdf.Name('/DeviceGray') if grayscale else pikepdf.Name('/DeviceRGB')
                
                elif mode == 'safe':
                    if '/SMask' in obj or '/Mask' in obj: continue
                    jpeg_bytes = jpeg_bytes_from_pil(pil_img, quality_val)
                    if original_size and len(jpeg_bytes) >= original_size: continue
                    obj.write(jpeg_bytes)
                    obj['/Filter'] = pikepdf.Name('/DCTDecode')

            except Exception: continue
        
        pdf.save(output_path)
        pdf.close()
        return True, "Success"
    except Exception as e: return False, f"Error: {e}"
    finally:
        for p in [temp_repaired, temp_cleaned]:
            if os.path.exists(p): os.remove(p)

# --- STREAMLIT UI ---
st.set_page_config(page_title="PDF Crusher", layout="centered")
st.title("🛡️ The PDF Link Remover & Compressor")

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

with st.sidebar:
    st.header("Configurations")
    mode_choice = st.selectbox("Mode", ["Lossless-Smart", "Safe Compression", "Aggressive Compression", "Rasterize (Standard)", "Rasterize (B&W Fax Mode)"])
    quality = st.slider("Quality/Zoom", 10, 100, 75)
    grayscale = st.checkbox("Convert to Grayscale")
    watermark = st.text_input("Text to Redact", "")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.button("PROCESS PDF"):
        with st.spinner("Processing... stop being impatient."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                tmp_in.write(uploaded_file.read())
                in_path = tmp_in.name
            
            out_path = in_path + "_out.pdf"
            mode_map = {
                "Safe Compression": 'safe', "Aggressive Compression": 'aggressive',
                "Rasterize (Standard)": 'rasterize', "Rasterize (B&W Fax Mode)": 'rasterize_fax',
                "Lossless-Smart": 'lossless-smart'
            }
            
            success, msg = process_pdf_pipeline(in_path, out_path, watermark, quality, mode_map[mode_choice], grayscale)
            
            if success:
                with open(out_path, "rb") as f:
                    st.session_state.processed_data = f.read()
                    st.session_state.file_name = f"processed_{uploaded_file.name}"
                st.success(msg)
            else:
                st.error(msg)
            
            if os.path.exists(in_path): os.remove(in_path)
            if os.path.exists(out_path): os.remove(out_path)

if st.session_state.processed_data:
    st.download_button(
        label="📥 DOWNLOAD NOW",
        data=st.session_state.processed_data,
        file_name=st.session_state.file_name,
        mime="application/pdf"
    )
