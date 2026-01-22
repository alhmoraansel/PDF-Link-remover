import streamlit as st
import os
import io
import tempfile
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from pikepdf import Name
from PIL import Image, ImageEnhance  # Added ImageEnhance

# Allow loading truncated images
Image.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. CORE LOGIC & HELPERS
# ==========================================

def get_raw_stream_length(obj):
    """Returns the actual compressed length of the stream on disk."""
    try:
        if '/Length' in obj:
            return int(obj['/Length'])
    except:
        pass
    try:
        return len(obj.read_raw_bytes())
    except:
        pass
    return 0

def pil_from_pdfimage(obj):
    """Safely extract PIL image from PDF Image XObject."""
    try:
        pdf_img = PdfImage(obj)
        return pdf_img.as_pil_image()
    except Exception:
        return None

def resize_image(pil_img, max_dim=None):
    if not max_dim:
        return pil_img
    
    width, height = pil_img.size
    if width <= max_dim and height <= max_dim:
        return pil_img

    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
        
    return pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def remove_text_watermark_fitz(input_path, output_path, text_to_remove):
    """
    Uses PyMuPDF to find text.
    """
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
    except Exception as e:
        return False, str(e)

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False, progress_queue=None):
    """
    Uses PyMuPDF (fitz) to render pages as images and rebuild the PDF.
    """
    try:
        src_doc = fitz.open(input_pdf_path)
        out_doc = fitz.open()

        if fax_mode:
            zoom = 2.5 
        else:
            zoom = 1.0 + (quality / 100.0)
            if zoom > 2.0: zoom = 2.0

        mat = fitz.Matrix(zoom, zoom)
        total_pages = len(src_doc)
        
        for i, page in enumerate(src_doc):
            if progress_queue:
                progress_queue.put(('progress', i + 1, total_pages))

            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB"
            if pix.n == 1: mode = "L"
            elif pix.n == 4: mode = "CMYK"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()

            if fax_mode:
                # --- UPDATED LOGIC START ---
                # 1. Ensure Grayscale
                if img.mode != 'L':
                    img = img.convert('L')

                # 2. Enhance Contrast (Separates faint text from grey background)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)  # Boost contrast by 2x
                
                # 3. Apply Threshold
                # Previous logic: < 200 became Black (0). This turned light grey (190) into black.
                # New logic: < 128 becomes Black. This keeps light grey (130+) as White (255).
                img = img.point(lambda x: 0 if x < 128 else 255, '1')
                
                img.save(buf, format="TIFF", compression="group4")
                # --- UPDATED LOGIC END ---
            else:
                if grayscale:
                    img = img.convert('L')
                elif mode == 'RGB':
                    try:
                        sample = img.resize((64, 64), resample=Image.NEAREST)
                        if sample.mode == 'RGB':
                            extrema = sample.getextrema()
                            if len(extrema) == 3 and extrema[0] == extrema[1] == extrema[2]:
                                img = img.convert('L')
                    except: pass

                img.save(buf, format="JPEG", quality=int(quality), optimize=True, progressive=True, subsampling=2)

            img_bytes = buf.getvalue()
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes)

        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close()
        out_doc.close()
        return True, "Rasterization successful"
    except Exception as e:
        return False, f"Rasterization failed: {e}"

def yield_images_from_resources(resources, processed_oids):
    """Recursively find images in a resource dictionary."""
    if '/XObject' not in resources:
        return

    xobjs = resources['/XObject']
    for name, xobj_ref in xobjs.items():
        try:
            xobj = xobj_ref 
            oid = getattr(xobj, 'objid', None)
            if oid is not None and oid in processed_oids:
                continue
            
            subtype = xobj.get('/Subtype')
            
            if subtype == pikepdf.Name('/Image'):
                if oid is not None:
                    processed_oids.add(oid)
                yield xobj
                
            elif subtype == pikepdf.Name('/Form'):
                if oid is not None:
                    processed_oids.add(oid)
                if '/Resources' in xobj:
                    yield from yield_images_from_resources(xobj['/Resources'], processed_oids)
        except Exception:
            continue

# ==========================================
# 2. MAIN PIPELINE
# ==========================================

def process_pdf_pipeline(args):
    """
    Core pipeline logic. 
    args: (input_path, output_path, quality_val, mode, grayscale, watermark_text, progress_queue)
    """
    input_path, output_path, quality_val, mode, grayscale, watermark_text, progress_queue = args
    
    temp_cleaned_path = None

    try:
        # --- Pre-processing: Watermark Removal ---
        current_input = input_path
        
        if watermark_text and watermark_text.strip():
            fd, temp_cleaned_path = tempfile.mkstemp(suffix="_wm.pdf")
            os.close(fd)
            
            if progress_queue:
                progress_queue.put(('status', f"DEBUG: Starting Watermark Removal for '{watermark_text}'...", 0))
            
            success, msg = remove_text_watermark_fitz(input_path, temp_cleaned_path, watermark_text)
            if success:
                current_input = temp_cleaned_path
                if progress_queue:
                    progress_queue.put(('status', "DEBUG: Watermark removal phase complete.", 0))
        
        # --- Branch 1: Rasterization ---
        if mode.startswith('rasterize'):
            if progress_queue:
                progress_queue.put(('status', f"DEBUG: Starting Rasterization (Mode: {mode})...", 0))
                
            is_fax = (mode == 'rasterize_fax')
            success, msg = rasterize_and_rebuild(
                current_input, 
                output_path, 
                quality_val, 
                grayscale, 
                fax_mode=is_fax,
                progress_queue=progress_queue
            )
            return success, msg

        # --- Branch 2: Structural Cleaning & Compression ---
        if progress_queue:
            progress_queue.put(('status', "DEBUG: Opening PDF for Structural Cleaning (Pikepdf)...", 0))

        pdf = pikepdf.open(current_input, allow_overwriting_input=True)
        
        # 1. Structural Cleaning
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

        # 2. Image Compression
        max_dim = 1500 if mode == 'aggressive' else None
        total_pages = len(pdf.pages)
        processed_oids = set() 

        for page_idx, page in enumerate(pdf.pages):
            # Report Progress
            if progress_queue:
                progress_queue.put(('progress', page_idx + 1, total_pages))

            if '/Resources' not in page:
                continue

            for obj in yield_images_from_resources(page['/Resources'], processed_oids):
                try:
                    # STRICT SKIP LOGIC
                    has_mask = False
                    for k in ['/Mask', '/SMask', '/ImageMask', '/Matte']:
                        if k in obj:
                            has_mask = True
                            break
                    if has_mask: continue
                    if '/Decode' in obj: continue

                    cs = obj.get('/ColorSpace')
                    is_safe_cs = False
                    if cs == pikepdf.Name('/DeviceRGB') or cs == pikepdf.Name('/DeviceGray'):
                        is_safe_cs = True
                    if not is_safe_cs: continue

                    current_filter = obj.get('/Filter')
                    if current_filter:
                        filters = current_filter if isinstance(current_filter, list) else [current_filter]
                        skip_filter = False
                        for f in filters:
                            if f in (pikepdf.Name('/CCITTFaxDecode'), pikepdf.Name('/JBIG2Decode'), pikepdf.Name('/JPXDecode')):
                                skip_filter = True
                                break
                        if skip_filter: continue
                    
                    if obj.get('/BitsPerComponent') == 1: continue

                    # EXTRACTION
                    img = pil_from_pdfimage(obj)
                    if img is None: continue
                    if img.mode in ('1', 'CMYK', 'P'): continue
                    if img.mode not in ('RGB', 'L'): continue

                    original_compressed_size = get_raw_stream_length(obj)

                    # PROCESSING
                    if grayscale:
                        if img.mode != "L":
                            new_img = img.convert("L")
                            new_mode = "DeviceGray"
                        else:
                            new_img = img
                            new_mode = "DeviceGray"
                    else:
                        if img.mode == "L":
                            new_img = img
                            new_mode = "DeviceGray"
                        else:
                            new_img = img.convert("RGB")
                            new_mode = "DeviceRGB"

                    if max_dim:
                        new_img = resize_image(new_img, max_dim)

                    # RECOMPRESSION CHECK
                    new_data = None
                    is_lossless = (mode == 'lossless-smart')
                    
                    if is_lossless:
                        buf = io.BytesIO()
                        new_img.save(buf, format="PNG", optimize=True)
                        temp_data = buf.getvalue()
                        if original_compressed_size > 0 and len(temp_data) < original_compressed_size:
                            new_data = temp_data
                            new_filter = pikepdf.Name("/FlateDecode")
                    else:
                        q = quality_val
                        sub = 0 if mode == 'safe' else 2
                        buf = io.BytesIO()
                        new_img.save(buf, format="JPEG", quality=q, subsampling=sub, optimize=True)
                        temp_data = buf.getvalue()
                        if original_compressed_size > 0 and len(temp_data) < original_compressed_size:
                            new_data = temp_data
                            new_filter = pikepdf.Name("/DCTDecode")

                    # WRITE BACK
                    if new_data:
                        obj.write(new_data, filter=new_filter)
                        obj["/Type"] = pikepdf.Name("/XObject")
                        obj["/Subtype"] = pikepdf.Name("/Image")
                        obj["/Width"] = new_img.width
                        obj["/Height"] = new_img.height
                        obj["/ColorSpace"] = pikepdf.Name("/" + new_mode)
                        obj["/BitsPerComponent"] = 8
                        obj["/Length"] = len(new_data)
                        
                        current_keys = list(obj.keys())
                        whitelist = {'/Type', '/Subtype', '/Width', '/Height', 
                                     '/ColorSpace', '/BitsPerComponent', '/Length', '/Filter'}
                        for k in current_keys:
                            if k not in whitelist:
                                del obj[k]

                except Exception as e:
                    continue

        pdf.save(output_path, object_stream_mode=pikepdf.ObjectStreamMode.generate)
        pdf.close()
        return True, "Success"

    except Exception as e:
        return False, f"Error: {e}"
    finally:
        if temp_cleaned_path and os.path.exists(temp_cleaned_path):
            try: os.remove(temp_cleaned_path)
            except: pass

# ==========================================
# 3. STREAMLIT UTILS & UI
# ==========================================

class StreamlitLogger:
    """Simulates the multiprocessing queue but writes directly to Streamlit."""
    def __init__(self, log_placeholder, progress_bar):
        self.log_placeholder = log_placeholder
        self.progress_bar = progress_bar

    def put(self, msg):
        # msg structure: ('progress', current, total) or ('status', text, 0)
        msg_type = msg[0]
        if msg_type == 'progress':
            _, current, total = msg
            if total > 0:
                percent = current / total
                self.progress_bar.progress(percent)
        elif msg_type == 'status':
            _, text, _ = msg
            self.log_placeholder.code(text)

def main():
    st.set_page_config(page_title="CleanPDF Hybrid (Streamlit)", layout="centered")

    st.title("📄 CleanPDF (Hybrid) - Streamlit Edition")
    st.write("Identical core logic to the original tool, running in the browser.")

    # --- UI Section: File Upload ---
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    # --- UI Section: Configuration ---
    with st.expander("⚙️ Configuration", expanded=True):
        
        # Watermark Section
        st.subheader("Watermark Removal")
        enable_wm = st.checkbox("Remove Specific Text")
        wm_text = ""
        if enable_wm:
            wm_text = st.text_input("Text to remove (comma separated)", placeholder="Confidential, Draft, Do Not Copy")

        st.markdown("---")
        
        # Compression Section
        st.subheader("Optimization")
        enable_compress = st.checkbox("Enable Image Compression")
        
        mode = "safe"
        quality = 100
        grayscale = False
        
        if enable_compress:
            mode_display = st.selectbox("Compression Mode", [
                "Safe Compression", 
                "Aggressive Compression", 
                "Lossless Smart",
                "Rasterize (Standard)",
                "Rasterize (B&W Fax Mode)"
            ])
            
            # Map display string back to internal mode keys
            mode_map = {
                "Safe Compression": "safe",
                "Aggressive Compression": "aggressive",
                "Lossless Smart": "lossless-smart",
                "Rasterize (Standard)": "rasterize",
                "Rasterize (B&W Fax Mode)": "rasterize_fax"
            }
            mode = mode_map[mode_display]
            
            if "Rasterize" in mode_display or mode_display in ["Safe Compression", "Aggressive Compression"]:
                quality = st.slider("Quality %", 10, 100, 75)
            
            grayscale = st.checkbox("Convert Images to Grayscale")

    # --- Processing Block ---
    if st.button("Start Processing", type="primary", disabled=not uploaded_files):
        
        progress_bar = st.progress(0)
        log_area = st.empty()  # Placeholder for debug messages
        
        results = []

        # Create our logger bridge
        logger = StreamlitLogger(log_area, progress_bar)

        for up_file in uploaded_files:
            logger.put(('status', f"Processing file: {up_file.name}...", 0))
            
            # 1. Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                tmp_in.write(up_file.getvalue())
                input_path = tmp_in.name

            # 2. Prepare output path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
                output_path = tmp_out.name

            # 3. RUN PIPELINE
            args = (input_path, output_path, quality, mode, grayscale, wm_text, logger)
            
            success, msg = process_pdf_pipeline(args)

            if success:
                logger.put(('status', f"Successfully processed: {up_file.name}", 0))
                with open(output_path, "rb") as f:
                    pdf_bytes = f.read()
                
                results.append({
                    "name": f"cleaned_{up_file.name}",
                    "data": pdf_bytes
                })
            else:
                logger.put(('status', f"ERROR processing {up_file.name}: {msg}", 0))
                st.error(f"Error in {up_file.name}: {msg}")

            # Cleanup
            try: os.remove(input_path)
            except: pass
            try: os.remove(output_path)
            except: pass
            
        progress_bar.progress(100)
        st.success("Batch Processing Complete!")

        # --- Download Section ---
        st.subheader("📥 Download Results")
        for res in results:
            st.download_button(
                label=f"Download {res['name']}",
                data=res['data'],
                file_name=res['name'],
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
