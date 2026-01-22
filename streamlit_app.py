import streamlit as st
import os
import io
import tempfile
import time
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from PIL import Image

# --- DEBUG UTILITY ---
def debug_log(message, is_error=False):
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg) # Goes to Streamlit Cloud Logs
    if is_error:
        st.error(msg)
    else:
        st.info(msg)

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
    return pil_img.convert('RGB')

def jpeg_bytes_from_pil(pil_img, quality, subsampling=None):
    buf = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None: save_kwargs["subsampling"] = int(subsampling)
    pil_img.save(buf, **save_kwargs)
    return buf.getvalue()

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False):
    try:
        debug_log(f"Starting Rasterization (Fax={fax_mode})")
        src_doc = fitz.open(input_pdf_path)
        out_doc = fitz.open()
        zoom = 2.5 if fax_mode else (1.0 + (quality / 100.0))
        if zoom > 2.0: zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        
        for i, page in enumerate(src_doc):
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
            if i % 5 == 0: debug_log(f"Rasterized {i} pages...")
            
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close(); out_doc.close()
        return True, "Rasterization complete"
    except Exception as e: return False, str(e)

def process_pdf_pipeline(input_path, output_path, watermark_text, quality_val, mode, grayscale):
    debug_log(f"Pipeline started. Mode: {mode}")
    fd1, temp_rep = tempfile.mkstemp(suffix=".pdf"); os.close(fd1)
    fd2, temp_cln = tempfile.mkstemp(suffix=".pdf"); os.close(fd2)
    
    try:
        debug_log("Repairing PDF metadata...")
        with pikepdf.open(input_path) as pdf:
            pdf.save(temp_rep, fix_metadata_version=True)
        
        debug_log("Cleaning links and redacting...")
        doc = fitz.open(temp_rep)
        for page in doc:
            if watermark_text:
                for rect in page.search_for(watermark_text): page.add_redact_annot(rect)
                page.apply_redactions(images=0, graphics=0)
            page.clean_contents()
        doc.save(temp_cln, garbage=4, deflate=True)
        doc.close()

        if 'rasterize' in mode:
            return rasterize_and_rebuild(temp_cln, output_path, quality_val, grayscale, mode=='rasterize_fax')

        debug_log("Running Image Compression Loop...")
        with pikepdf.open(temp_cln) as pdf:
            pdf.docinfo.clear()
            image_count = 0
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
                        image_count += 1
                    except: continue
            debug_log(f"Compressed {image_count} images.")
            pdf.save(output_path)
        return True, "Pipeline Success"
    except Exception as e:
        debug_log(f"CRITICAL ERROR in pipeline: {str(e)}", is_error=True)
        return False, str(e)
    finally:
        for p in [temp_rep, temp_cln]:
            if os.path.exists(p): os.remove(p)

# --- STREAMLIT INTERFACE ---
st.set_page_config(page_title="PDF DEBUGGER", layout="wide")
st.title("🛡️ PDF Processor (DEBUG MODE)")

# Persistent state
if "out_bytes" not in st.session_state: st.session_state.out_bytes = None
if "out_name" not in st.session_state: st.session_state.out_name = ""

with st.sidebar:
    st.header("Settings")
    mode_choice = st.selectbox("Mode", ["lossless-smart", "safe", "aggressive", "rasterize", "rasterize_fax"])
    quality = st.slider("Quality", 10, 100, 70)
    grayscale = st.checkbox("Grayscale")
    watermark = st.text_input("Redact Text", "")
    if st.button("Reset Session"):
        st.session_state.out_bytes = None
        st.rerun()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    debug_log(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    if st.button("PROCESS PDF"):
        debug_log("Process button clicked.")
        # Clear previous results
        st.session_state.out_bytes = None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
            tmp_in.write(uploaded_file.getbuffer())
            in_path = tmp_in.name
            debug_log(f"Temporary input file created at: {in_path}")
        
        out_path = in_path + "_out.pdf"
        
        success, msg = process_pdf_pipeline(in_path, out_path, watermark, quality, mode_choice, grayscale)
        
        if success:
            debug_log(f"Pipeline returned success. Reading {out_path} into memory...")
            try:
                with open(out_path, "rb") as f:
                    st.session_state.out_bytes = f.read()
                st.session_state.out_name = f"processed_{uploaded_file.name}"
                debug_log(f"Stored {len(st.session_state.out_bytes)} bytes in session state.")
                st.success("PROCESSING DONE. DOWNLOAD BUTTON SHOULD APPEAR BELOW.")
            except Exception as e:
                debug_log(f"Failed to read output file: {e}", is_error=True)
        else:
            st.error(f"Pipeline failed: {msg}")
        
        # Cleanup
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)

st.divider()

if st.session_state.out_bytes is not None:
    debug_log(f"Rendering Download Button for {st.session_state.out_name}...")
    st.download_button(
        label="📥 DOWNLOAD NOW (CLICK ONCE)",
        data=st.session_state.out_bytes,
        file_name=st.session_state.out_name,
        mime="application/pdf",
        on_click=lambda: debug_log("DOWNLOAD BUTTON TRIGGERED BY USER")
    )
else:
    st.warning("No processed data found in session state. Process a file first.")
