import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import io
import tempfile
import shutil
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from pikepdf import Name, Dictionary, Array
from PIL import Image, ImageOps
import concurrent.futures
import multiprocessing
import queue  # Standard queue for polling

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
    """
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
    elif pil_img.mode != 'RGB':
        return pil_img.convert('RGB')
    return pil_img

def read_stream_bytes(obj):
    try:
        return obj.get_stream_buffer()
    except Exception:
        pass
    try:
        return obj.read_bytes()
    except Exception:
        pass
    return b''

# ----------------------------
# Rasterization Logic
# ----------------------------

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False, progress_queue=None):
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
# Main Pipeline (Robust Repair -> Clean -> Compress)
# ----------------------------

def process_pdf_pipeline(args):
    # Unpack arguments from threading wrapper
    input_path, output_path, watermark_text, quality_val, mode, grayscale, progress_queue = args
    
    # Define temp file paths
    fd1, temp_repaired = tempfile.mkstemp(suffix="_repaired.pdf")
    os.close(fd1)
    fd2, temp_cleaned = tempfile.mkstemp(suffix="_cleaned.pdf")
    os.close(fd2)

    try:
        if progress_queue:
            progress_queue.put(('status', "Step 1/3: Repairing Structure...", 0))

        # --- Step 1: Repair PDF (Pikepdf) ---
        try:
            pdf = pikepdf.open(input_path, allow_overwriting_input=True)
            pdf.save(temp_repaired, fix_metadata_version=True)
            pdf.close()
        except Exception as e:
            shutil.copy2(input_path, temp_repaired)

        # --- Step 2: Clean Watermarks/Links (PyMuPDF) ---
        if progress_queue:
            progress_queue.put(('status', "Step 2/3: Cleaning Content...", 0))

        try:
            doc = fitz.open(temp_repaired)
            doc.set_metadata({}) 
            total_pages = len(doc)

            phrases = []
            if watermark_text and watermark_text.strip():
                phrases = [p.strip() for p in watermark_text.split(',')]

            for i, page in enumerate(doc):
                # Optional: Update progress for cleaning massive docs
                if i % 10 == 0 and progress_queue:
                     progress_queue.put(('status', f"Cleaning Page {i+1}/{total_pages}...", 0))

                # Remove Links
                try:
                    for link in page.get_links():
                        page.delete_link(link)
                except: pass

                # Remove Annotations
                try:
                    for annot in list(page.annots()):
                        if annot.type[0] == 1: 
                            page.delete_annot(annot)
                except: pass

                # Redact Text
                if phrases:
                    try:
                        for phrase in phrases:
                            if not phrase: continue
                            hits = page.search_for(phrase)
                            for rect in hits:
                                page.add_redact_annot(rect)
                        
                        # REDACT FLAGS: images=0 (keep), graphics=0 (keep), text=1 (remove)
                        page.apply_redactions(images=0, graphics=0)
                    except: pass

                try: page.clean_contents()
                except: pass

            doc.save(temp_cleaned, garbage=4, deflate=True)
            doc.close()
            current_input = temp_cleaned
        except Exception as e:
            current_input = temp_repaired
        
        # --- Step 3: Compression / Rasterization ---
        if progress_queue:
            progress_queue.put(('status', "Step 3/3: Optimizing...", 0))

        # Branch A: Rasterization
        if mode.startswith('rasterize'):
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

        # Branch B: Object-based Optimization (Pikepdf)
        try:
            pdf = pikepdf.open(current_input, allow_overwriting_input=True)
            pdf.docinfo.clear()

            # Clean Structure
            try:
                if '/Outlines' in pdf.Root: del pdf.Root['/Outlines']
            except: pass
            
            # Iterate objects (Standard Pikepdf approach for max compression)
            max_dim = 1500 if mode == 'aggressive' else None
            
            # Count objects for progress estimation (rough)
            all_objects = list(pdf.objects)
            total_objs = len(all_objects)
            
            for idx, obj in enumerate(all_objects):
                # Throttle progress updates
                if idx % 50 == 0 and progress_queue:
                    progress_queue.put(('status', f"Optimizing Object {idx}/{total_objs}...", 0))

                try:
                    if not (isinstance(obj, pikepdf.Stream) and obj.get('/Subtype') == pikepdf.Name('/Image')):
                        continue
                    
                    # Size Guard
                    try:
                        if '/Width' in obj and int(obj['/Width']) < 50: continue
                        if '/Height' in obj and int(obj['/Height']) < 50: continue
                    except: pass

                    # Extract
                    try:
                        pdf_img = PdfImage(obj)
                        pil_img = pil_from_pdfimage(pdf_img)
                        if pil_img is None: continue
                    except: continue

                    # Original Size
                    original_size = get_raw_stream_length(obj)

                    # 1. Grayscale
                    if grayscale and pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')

                    # 2. Resize
                    if max_dim:
                        pil_img = resize_image(pil_img, max_dim)

                    # 3. Encode
                    new_data = None
                    is_lossless = (mode == 'lossless-smart')
                    
                    has_alpha = pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info)

                    if is_lossless:
                        png_data = png_bytes_from_pil(pil_img, optimize=True)
                        if original_size == 0 or len(png_data) < original_size:
                            new_data = png_data
                            new_filter = pikepdf.Name('/FlateDecode')
                            new_colorspace = pikepdf.Name('/DeviceRGB') if pil_img.mode == 'RGB' else pikepdf.Name('/DeviceGray')
                    else:
                        if has_alpha:
                            pil_img = flatten_alpha(pil_img)
                        if pil_img.mode not in ('RGB', 'L'):
                            pil_img = pil_img.convert('RGB')
                        
                        q = quality_val
                        sub = 0 if mode == 'safe' else 2
                        jpeg_data = jpeg_bytes_from_pil(pil_img, q, subsampling=sub)
                        
                        accept = False
                        if original_size == 0: accept = True
                        elif len(jpeg_data) < original_size: accept = True
                        elif mode == 'aggressive' and len(jpeg_data) < original_size * 1.1: accept = True
                        
                        if accept:
                            new_data = jpeg_data
                            new_filter = pikepdf.Name('/DCTDecode')
                            new_colorspace = pikepdf.Name('/DeviceRGB') if pil_img.mode == 'RGB' else pikepdf.Name('/DeviceGray')

                    if new_data:
                        obj.clear()
                        obj.write(new_data)
                        obj['/Type'] = pikepdf.Name('/XObject')
                        obj['/Subtype'] = pikepdf.Name('/Image')
                        obj['/Width'] = pil_img.width
                        obj['/Height'] = pil_img.height
                        obj['/ColorSpace'] = new_colorspace
                        obj['/BitsPerComponent'] = 8
                        obj['/Filter'] = new_filter
                        
                        for k in ['/Decode', '/Mask', '/SMask', '/Matte', '/ColorKeyMask', '/DecodeParms']:
                            if k in obj:
                                try: del obj[k]
                                except: pass

                except Exception:
                    continue

            pdf.save(output_path, object_stream_mode=pikepdf.ObjectStreamMode.generate)
            pdf.close()
            return True, "Success"
            
        except Exception as e:
            # If optimization fails, copy the cleaned file
            shutil.copy2(current_input, output_path)
            return True, f"Success (Optimization skipped: {e})"

    except Exception as e:
        return False, f"Error: {e}"
    finally:
        for p in [temp_repaired, temp_cleaned]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass

# ----------------------------
# Threading & UI Logic
# ----------------------------

def start_processing_thread(files_to_process, output_dir=None, single_output=None):
    # Disable UI
    for widget in [save_button, batch_button, browse_button, paste_wm_button]:
        widget.state(['disabled'])
        
    # Unbind click to prevent double loading during process
    input_entry.unbind("<Button-1>")
    
    progress_bar['value'] = 0
    status_label.config(text="Initializing...", foreground="#0056b3")

    # Gather Settings
    quality = int(quality_scale.get()) if compress_var.get() else 100
    grayscale = grayscale_var.get()
    wm_text = watermark_entry.get().strip()

    mode_choice = compression_mode_var.get()
    if mode_choice == "Safe Compression": mode = 'safe'
    elif mode_choice == "Aggressive Compression": mode = 'aggressive'
    elif mode_choice == "Rasterize (Standard)": mode = 'rasterize'
    elif mode_choice == "Rasterize (B&W Fax Mode)": mode = 'rasterize_fax'
    else: mode = 'lossless-smart'

    # Setup Queue
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    tasks = []
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        if single_output:
            current_output = single_output
        else:
            name, ext = os.path.splitext(filename)
            current_output = os.path.join(output_dir, f"{name}_cleaned{ext}")
        
        tasks.append((file_path, current_output, wm_text, quality, mode, grayscale, progress_queue))

    def poll_queue():
        try:
            while True:
                msg = progress_queue.get_nowait()
                msg_type = msg[0]
                if msg_type == 'progress':
                    _, current, total = msg
                    if total > 0:
                        percent = (current / total) * 100
                        progress_bar['value'] = percent
                        status_label.config(text=f"Processing Page {current}/{total}")
                elif msg_type == 'status':
                    _, text, _ = msg
                    status_label.config(text=text)
        except queue.Empty:
            pass
            
        if save_button.instate(['disabled']):
            root.after(100, poll_queue)

    root.after(100, poll_queue)

    def run_job():
        success_count = 0
        errors = []
        total_files = len(tasks)
        completed = 0

        max_workers = 1 # Avoid conflicts with heavy PDF ops
        
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
                
        root.after(0, lambda: finish_processing(success_count, total_files, errors))

    threading.Thread(target=run_job, daemon=True).start()

def finish_processing(success_count, total_count, errors):
    progress_bar['value'] = 100
    status_label.config(text="Processing Complete", foreground="green")
    
    for widget in [save_button, batch_button, browse_button, paste_wm_button]:
        widget.state(['!disabled'])
    
    # Rebind click
    input_entry.bind("<Button-1>", lambda e: open_file())

    result_msg = f"Processed {success_count}/{total_count} files successfully."
    if errors:
        result_msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showwarning("Completed with Issues", result_msg)
    else:
        messagebox.showinfo("Success", result_msg)

# --- UI Action Functions ---

def open_file(event=None):
    # If the widget is disabled (during processing), do nothing
    if input_entry.instate(['disabled']):
        return
        
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, file_path)
        folder, name = os.path.split(file_path)
        name_root, ext = os.path.splitext(name)
        output_entry.delete(0, tk.END)
        output_entry.insert(0, os.path.join(folder, f"{name_root}_cleaned{ext}"))
    return "break" # Stop default focus behavior

def paste_watermark_text():
    try:
        text = root.clipboard_get()
        if text:
            watermark_entry.delete(0, tk.END)
            watermark_entry.insert(0, text.strip())
            status_label.config(text="Watermark pasted", foreground="green")
            root.after(1500, lambda: status_label.config(text="Ready", foreground="gray"))
    except Exception as e:
        messagebox.showerror("Paste Error", f"Could not paste: {e}")

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
    root.title("CleanPDF")
    root.geometry("500x600") 
    root.minsize(500, 600)

    # 1. Styling
    style = ttk.Style()
    style.theme_use('clam')

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
    ttk.Label(header_frame, text="PDF Optimizer", font=("Segoe UI", 16, "bold"), foreground=FG_COLOR).pack(side=tk.LEFT)
    status_label = ttk.Label(header_frame, text="Ready", foreground="gray", font=("Segoe UI", 9))
    status_label.pack(side=tk.RIGHT, anchor="s")

    # Section 1: Files & Watermark
    file_frame = ttk.LabelFrame(main_container, text="Configuration", padding=(15, 10))
    file_frame.pack(fill=tk.X, pady=(0, 15))

    # Input (Clickable)
    ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
    input_entry = ttk.Entry(file_frame)
    input_entry.grid(row=0, column=1, sticky="ew", padx=5)
    input_entry.bind("<Button-1>", open_file) # Click to open
    
    browse_button = ttk.Button(file_frame, text="...", width=4, command=open_file)
    browse_button.grid(row=0, column=2, sticky="e")

    # Output
    ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky="w", pady=5)
    output_entry = ttk.Entry(file_frame)
    output_entry.grid(row=1, column=1, sticky="ew", padx=5, columnspan=2)

    # Watermark + Paste
    ttk.Label(file_frame, text="Watermark:").grid(row=2, column=0, sticky="w", pady=5)
    watermark_entry = ttk.Entry(file_frame)
    watermark_entry.grid(row=2, column=1, sticky="ew", padx=5)
    
    paste_wm_button = ttk.Button(file_frame, text="Paste", width=6, command=paste_watermark_text)
    paste_wm_button.grid(row=2, column=2, sticky="e")

    file_frame.columnconfigure(1, weight=1)

    # Section 2: Optimization
    opt_frame = ttk.LabelFrame(main_container, text="Optimization", padding=(15, 10))
    opt_frame.pack(fill=tk.X, pady=(0, 15))

    compress_var = tk.BooleanVar(value=False)
    compress_check = ttk.Checkbutton(opt_frame, text="Enable Compression & Rasterization", variable=compress_var, command=toggle_compression)
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
    grayscale_check = ttk.Checkbutton(settings_inner, text="Convert to Grayscale", variable=grayscale_var, state='disabled')
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