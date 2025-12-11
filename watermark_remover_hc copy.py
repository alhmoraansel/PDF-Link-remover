import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import io
import tempfile
import shutil
import fitz  # PyMuPDF (Restored for Rasterization)
import pikepdf
from pikepdf.models.image import PdfImage
from pikepdf import Name, Dictionary, Array
from PIL import Image, ImageOps
import concurrent.futures
import multiprocessing

# Allow loading truncated images
Image.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Helper / Compression Utilities
# ----------------------------

def get_raw_stream_length(obj):
    """
    Returns the actual compressed length of the stream on disk.
    """
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
    """
    Safely extract PIL image from PDF Image XObject.
    RETURNS RAW IMAGE. NO AUTO-CONVERSION.
    """
    try:
        # We use PdfImage because it handles internal PDF filters (Flate, etc.) better 
        # than raw byte reading, but we strictly do NOT convert the output.
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

def jpeg_bytes_from_pil(pil_img, quality, subsampling=None):
    buf = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None:
        save_kwargs["subsampling"] = int(subsampling)
    pil_img.save(buf, **save_kwargs)
    return buf.getvalue()

def png_bytes_from_pil(pil_img, optimize=True):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=bool(optimize))
    return buf.getvalue()

# ----------------------------
# Rasterization Logic (Using PyMuPDF/fitz)
# ----------------------------

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False):
    """
    Uses PyMuPDF (fitz) to render pages as images and rebuild the PDF.
    This effectively flattens all layers, annotations, and vector graphics.
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

                img.save(buf, 
                         format="JPEG", 
                         quality=int(quality), 
                         optimize=True, 
                         progressive=True, 
                         subsampling=2)

            img_bytes = buf.getvalue()
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes)

        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close()
        out_doc.close()
        return True, "Rasterization successful"
    except Exception as e:
        return False, f"Rasterization failed: {e}"

# ----------------------------
# Main pipeline (Worker Function)
# ----------------------------

def process_pdf_pipeline(args):
    input_path, output_path, quality_val, mode, grayscale = args
    
    try:
        # --- Branch 1: Rasterization (Uses PyMuPDF/fitz) ---
        if mode.startswith('rasterize'):
            is_fax = (mode == 'rasterize_fax')
            success, msg = rasterize_and_rebuild(input_path, output_path, quality_val, grayscale, fax_mode=is_fax)
            return success, msg

        # --- Branch 2: Structural Cleaning & Compression (Uses Pikepdf) ---
        pdf = pikepdf.open(input_path, allow_overwriting_input=True)
        
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

        # 2. Image Compression / Optimization
        max_dim = 1500 if mode == 'aggressive' else None

        for obj in list(pdf.objects):
            try:
                # Validation: Must be an Image XObject
                if not (isinstance(obj, pikepdf.Stream) and obj.get('/Subtype') == pikepdf.Name('/Image')):
                    continue
                
                # =========================================================
                # THE GATEKEEPER: STRICT SKIP LOGIC
                # =========================================================

                # 1. Check for Masks (Alpha/Stencil/Transparency)
                # If these exist, we CANNOT simply replace the stream with a JPEG.
                has_mask = False
                for k in ['/Mask', '/SMask', '/ImageMask', '/Matte']:
                    if k in obj:
                        has_mask = True
                        break
                if has_mask:
                    # print(f"[SKIP] Mask detected on object {obj.objid}")
                    continue

                # 2. Check for Decode Arrays
                # If /Decode exists (e.g. [1 0] for inversion), we skip. 
                # Converting to standard JPEG + removing Decode = Inverted Image (Black Box).
                if '/Decode' in obj:
                    # print(f"[SKIP] Decode array detected on object {obj.objid}")
                    continue

                # 3. Check for Dangerous ColorSpaces
                # We only touch simple DeviceRGB or DeviceGray.
                # ICCBased, Indexed, Separation, Lab, CMYK = SKIP.
                cs = obj.get('/ColorSpace')
                is_safe_cs = False
                if cs == pikepdf.Name('/DeviceRGB') or cs == pikepdf.Name('/DeviceGray'):
                    is_safe_cs = True
                
                if not is_safe_cs:
                    # print(f"[SKIP] Unsafe ColorSpace {cs} on object {obj.objid}")
                    continue

                # 4. Check for 1-bit or Text Filters
                # JBIG2/CCITT are for text. Converting to JPEG bloats size and ruins edge sharpness.
                current_filter = obj.get('/Filter')
                if current_filter:
                    filters = current_filter if isinstance(current_filter, list) else [current_filter]
                    skip_filter = False
                    for f in filters:
                        if f in (pikepdf.Name('/CCITTFaxDecode'), pikepdf.Name('/JBIG2Decode'), pikepdf.Name('/JPXDecode')):
                            skip_filter = True
                            break
                    if skip_filter:
                        # print(f"[SKIP] Text filter detected on object {obj.objid}")
                        continue
                
                if obj.get('/BitsPerComponent') == 1:
                    # print(f"[SKIP] 1-bit image detected on object {obj.objid}")
                    continue

                # =========================================================
                # EXTRACTION & PROCESSING
                # =========================================================

                img = pil_from_pdfimage(obj)
                if img is None:
                    continue

                # 5. Last Line of Defense: PIL Mode Check
                # If PIL says it's 1-bit, CMYK, or Palette, we bail.
                if img.mode in ('1', 'CMYK', 'P'):
                    # print(f"[SKIP] Unsafe PIL Mode {img.mode} on object {obj.objid}")
                    continue
                
                # Check safe modes
                if img.mode not in ('RGB', 'L'):
                    continue

                # Capture original size
                original_compressed_size = get_raw_stream_length(obj)

                # Process Image
                # Normalize to standard RGB or Grayscale
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

                # Generate Candidate Stream
                new_data = None
                is_lossless = (mode == 'lossless-smart')
                
                if is_lossless:
                     buf = io.BytesIO()
                     new_img.save(buf, format="PNG", optimize=True)
                     temp_data = buf.getvalue()
                     # STRICT Size Check
                     if original_compressed_size > 0 and len(temp_data) < original_compressed_size:
                         new_data = temp_data
                         new_filter = pikepdf.Name("/FlateDecode")
                else:
                    q = quality_val
                    sub = 0 if mode == 'safe' else 2
                    buf = io.BytesIO()
                    new_img.save(buf, format="JPEG", quality=q, subsampling=sub, optimize=True)
                    temp_data = buf.getvalue()
                    
                    # STRICT Size Check
                    if original_compressed_size > 0 and len(temp_data) < original_compressed_size:
                         new_data = temp_data
                         new_filter = pikepdf.Name("/DCTDecode")

                # =========================================================
                # SCORCHED EARTH WRITER
                # =========================================================
                
                if new_data:
                    # We are replacing the stream. We must PURGE any metadata that might 
                    # conflict with our new standard JPEG/PNG stream.
                    
                    # 1. Update the stream content
                    obj.write(new_data, filter=new_filter)
                    
                    # 2. Define the new Valid Keys
                    # These are the ONLY keys that should exist for a standard Image XObject
                    obj["/Type"] = pikepdf.Name("/XObject")
                    obj["/Subtype"] = pikepdf.Name("/Image")
                    obj["/Width"] = new_img.width
                    obj["/Height"] = new_img.height
                    obj["/ColorSpace"] = pikepdf.Name("/" + new_mode)
                    obj["/BitsPerComponent"] = 8
                    obj["/Length"] = len(new_data)
                    
                    # 3. NUKE everything else
                    # We iterate over a copy of keys to avoid modification issues
                    current_keys = list(obj.keys())
                    
                    # Whitelist of keys we explicitly set or allow
                    whitelist = {
                        '/Type', '/Subtype', 
                        '/Width', '/Height', 
                        '/ColorSpace', '/BitsPerComponent', 
                        '/Length', '/Filter'
                    }

                    for k in current_keys:
                        if k not in whitelist:
                            # print(f"[CLEAN] Removing key {k} from object {obj.objid}")
                            del obj[k]

            except Exception as e:
                # print(f"[ERROR] {e}")
                continue

        pdf.save(output_path, object_stream_mode=pikepdf.ObjectStreamMode.generate)
        pdf.close()
        return True, "Success"

    except Exception as e:
        return False, f"Error: {e}"

# ----------------------------
# Threading & UI Logic
# ----------------------------

def start_processing_thread(files_to_process, output_dir=None, single_output=None):
    # Disable UI
    for widget in [save_button, batch_button, browse_button]:
        widget.state(['disabled'])
    
    progress_bar['value'] = 0
    status_label.config(text="Initializing...", foreground="#0056b3")

    quality = int(quality_scale.get()) if compress_var.get() else 100
    grayscale = grayscale_var.get()

    mode_choice = compression_mode_var.get()
    if mode_choice == "Safe Compression": mode = 'safe'
    elif mode_choice == "Aggressive Compression": mode = 'aggressive'
    elif mode_choice == "Rasterize (Standard)": mode = 'rasterize'
    elif mode_choice == "Rasterize (B&W Fax Mode)": mode = 'rasterize_fax'
    else: mode = 'lossless-smart'

    tasks = []
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        if single_output:
            current_output = single_output
        else:
            name, ext = os.path.splitext(filename)
            current_output = os.path.join(output_dir, f"{name}_cleaned{ext}")
        
        tasks.append((file_path, current_output, quality, mode, grayscale))

    def run_job():
        success_count = 0
        errors = []
        total_files = len(tasks)
        completed = 0

        max_workers = multiprocessing.cpu_count()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_pdf_pipeline, task): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                try:
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        errors.append(f"{filename}: {msg}")
                except Exception as exc:
                    errors.append(f"{filename}: Exception: {exc}")
                
                completed += 1
                percentage = (completed / total_files) * 100
                root.after(0, lambda p=percentage, f=filename: update_ui_progress(p, f))

        root.after(0, lambda: finish_processing(success_count, total_files, errors))

    threading.Thread(target=run_job, daemon=True).start()

def update_ui_progress(val, filename):
    progress_bar.configure(value=val)
    status_label.config(text=f"Finished: {filename}")

def finish_processing(success_count, total_count, errors):
    progress_bar['value'] = 100
    status_label.config(text="Processing Complete", foreground="green")
    
    for widget in [save_button, batch_button, browse_button]:
        widget.state(['!disabled'])

    result_msg = f"Processed {success_count}/{total_count} files successfully."
    if errors:
        result_msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showwarning("Completed with Issues", result_msg)
    else:
        messagebox.showinfo("Success", result_msg)

# --- UI Action Functions ---

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, file_path)
        folder, name = os.path.split(file_path)
        name_root, ext = os.path.splitext(name)
        output_entry.delete(0, tk.END)
        output_entry.insert(0, os.path.join(folder, f"{name_root}_cleaned{ext}"))

def run_single_file():
    input_path = input_entry.get()
    output_path = output_entry.get()
    if not input_path:
        messagebox.showwarning("Input Missing", "Please select an input PDF.")
        return
    if not output_path:
        output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if not output_path: return
    start_processing_thread([input_path], single_output=output_path)

def run_batch():
    files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")], title="Select Multiple PDFs")
    if not files: return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir: return
    start_processing_thread(files, output_dir=output_dir)

def update_scale_label(val):
    quality_label_var.set(f"Quality: {int(float(val))}%")

def toggle_compression():
    if compress_var.get():
        quality_scale.state(['!disabled'])
        compression_mode_dropdown.state(['!disabled'])
        compression_mode_dropdown.config(state="readonly")
        grayscale_check.state(['!disabled'])
    else:
        quality_scale.state(['disabled'])
        compression_mode_dropdown.state(['disabled'])
        grayscale_check.state(['disabled'])

# ----------------------------
# GUI Setup
# ----------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root = tk.Tk()
    root.title("CleanPDF (Hybrid) - Fixed")
    root.geometry("500x550") 
    root.minsize(500, 550)

    # 1. Styling
    style = ttk.Style()
    style.theme_use('clam')

    # Colors
    BG_COLOR = "#ffffff"
    FG_COLOR = "#333333"
    ACCENT_COLOR = "#0066cc" 
    ACCENT_HOVER = "#0052a3"

    style.configure(".", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 10))
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TLabelframe", background=BG_COLOR, padding=15)
    style.configure("TLabelframe.Label", background=BG_COLOR, foreground="#666666", font=("Segoe UI", 9, "bold"))

    style.configure("TButton", padding=6, relief="flat", background="#e1e1e1", borderwidth=0)
    style.map("TButton", background=[('active', '#d4d4d4')])

    style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="white", font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[('active', ACCENT_HOVER)])

    # 2. Layout
    main_container = ttk.Frame(root, padding=20)
    main_container.pack(fill=tk.BOTH, expand=True)

    # Header
    header_frame = ttk.Frame(main_container)
    header_frame.pack(fill=tk.X, pady=(0, 20))
    ttk.Label(header_frame, text="PDF Cleaner (Hybrid)", font=("Segoe UI", 16, "bold"), foreground=FG_COLOR).pack(side=tk.LEFT)
    status_label = ttk.Label(header_frame, text="Ready", foreground="gray", font=("Segoe UI", 9))
    status_label.pack(side=tk.RIGHT, anchor="s")

    # Section 1: Files
    file_frame = ttk.LabelFrame(main_container, text="Configuration", padding=(15, 10))
    file_frame.pack(fill=tk.X, pady=(0, 15))

    # Input
    ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
    input_entry = ttk.Entry(file_frame)
    input_entry.grid(row=0, column=1, sticky="ew", padx=5)
    browse_button = ttk.Button(file_frame, text="...", width=4, command=open_file)
    browse_button.grid(row=0, column=2, sticky="e")

    # Output
    ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky="w", pady=5)
    output_entry = ttk.Entry(file_frame)
    output_entry.grid(row=1, column=1, sticky="ew", padx=5, columnspan=2)

    file_frame.columnconfigure(1, weight=1)

    # Section 2: Optimization Settings
    opt_frame = ttk.LabelFrame(main_container, text="Optimization", padding=(15, 10))
    opt_frame.pack(fill=tk.X, pady=(0, 15))

    compress_var = tk.BooleanVar(value=False)
    compress_check = ttk.Checkbutton(opt_frame, text="Enable Image Compression", variable=compress_var, command=toggle_compression)
    compress_check.pack(anchor="w", pady=(0, 10))

    settings_inner = ttk.Frame(opt_frame)
    settings_inner.pack(fill=tk.X, padx=10)

    compression_mode_var = tk.StringVar(value="Safe Compression")
    compression_mode_dropdown = ttk.Combobox(settings_inner, textvariable=compression_mode_var, state="disabled",
                                             values=[
                                                 "Safe Compression", 
                                                 "Aggressive Compression", 
                                                 "Lossless Smart",
                                                 "Rasterize (Standard)",
                                                 "Rasterize (B&W Fax Mode)"
                                             ])
    compression_mode_dropdown.pack(fill=tk.X, pady=(0, 10))

    quality_label_var = tk.StringVar(value="Quality: 75%")
    ttk.Label(settings_inner, textvariable=quality_label_var, font=("Segoe UI", 9)).pack(anchor="w")
    quality_scale = ttk.Scale(settings_inner, from_=10, to=100, orient="horizontal", command=update_scale_label)
    quality_scale.set(75)
    quality_scale.pack(fill=tk.X, pady=(0, 10))
    quality_scale.state(['disabled'])

    grayscale_var = tk.BooleanVar(value=False)
    grayscale_check = ttk.Checkbutton(settings_inner, text="Convert Images to Grayscale", variable=grayscale_var, state='disabled')
    grayscale_check.pack(anchor="w")

    # Section 3: Actions
    action_frame = ttk.Frame(main_container)
    action_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    progress_bar = ttk.Progressbar(action_frame, orient="horizontal", mode="determinate")
    progress_bar.pack(fill=tk.X, pady=(0, 15))

    btn_grid = ttk.Frame(action_frame)
    btn_grid.pack(fill=tk.X)

    save_button = ttk.Button(btn_grid, text="Process Single File", style="Accent.TButton", command=run_single_file)
    save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    batch_button = ttk.Button(btn_grid, text="Batch Process", style="Accent.TButton", command=run_batch)
    batch_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    root.mainloop()